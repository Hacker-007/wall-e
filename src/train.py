from glob import glob
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoModel, AutoConfig
import wandb

from processing import EncodedData, get_tokens
from processing.find_intervals import RELEVANT_CLASSES
from processing.wandb import WandbConfiguration


class VideoAudioDataset(Dataset):
    def __init__(self, data_dir: str):
        """
        Dataset for loading pre-encoded .pt files.

        Args:
            data_dir (str): Directory containing encoded .pt files.
        """

        self.encoded_files = glob(os.path.join(data_dir, "*.pt"))
        # self.interval_ids = [fname[5:-3] for fname in self.encoded_files]

    def __len__(self):
        return len(self.encoded_files)

    def __getitem__(self, idx) -> EncodedData:
        return get_tokens(self.encoded_files[idx])


def collate(batch: list[EncodedData]):
    nouns, audio_tokens, video_tokens = zip(*batch)

    # Remove the batch dimension (1, sequence_length, 768) -> (sequence_length, 768)
    audio_tokens = [token.squeeze(0) for token in audio_tokens]
    video_tokens = [token.squeeze(0) for token in video_tokens]

    # Find max sequence lengths
    max_audio_length = max(token.size(0) for token in audio_tokens)
    max_video_length = max(token.size(0) for token in video_tokens)

    # Pad audio and video tokens
    padded_audio = torch.zeros((len(batch), max_audio_length, 768))
    padded_video = torch.zeros((len(batch), max_video_length, 768))
    audio_mask = torch.zeros((len(batch), max_audio_length), dtype=torch.long)
    video_mask = torch.zeros((len(batch), max_video_length), dtype=torch.long)
    for i in range(len(batch)):
        audio_len = audio_tokens[i].size(0)
        video_len = video_tokens[i].size(0)
        padded_audio[i, :audio_len] = audio_tokens[i]
        padded_video[i, :video_len] = video_tokens[i]
        audio_mask[i, :audio_len] = 1
        video_mask[i, :video_len] = 1

    # `nouns` must be remapped to the range [0, len(RELEVANT_CLASSES))
    # because this is the range that the loss function expects classes to
    # be in. This can be reversed when doing predictions with the operation
    # RELEVANT_CLASSES[predicted_noun].
    noun_mapping = {noun: idx for idx, noun in enumerate(RELEVANT_CLASSES)}
    nouns = [noun_mapping[noun] for noun in nouns]
    nouns = torch.tensor(nouns, dtype=torch.long)
    return nouns, padded_audio, padded_video, audio_mask, video_mask


class MultimodalTransformer(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased"):
        """
        Transformer model for processing audio and video tokens.

        Args:
            model_name (str): Name of pre-trained transformer.
        """
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, len(RELEVANT_CLASSES)),
        )

    def forward(self, audio_tokens, video_tokens, audio_mask, video_mask):
        """
        Forward pass through the model.

        Args:
            audio_tokens (torch.Tensor): Shape (batch, seq_len_a, hidden_size)
            video_tokens (torch.Tensor): Shape (batch, seq_len_v, hidden_size)
            audio_mask (torch.Tensor): Shape (batch, seq_len_a)
            video_mask (torch.Tensor): Shape (batch, seq_len_v)

        Returns:
            torch.Tensor: Logits for each class.
        """
        all_tokens = torch.cat((audio_tokens, video_tokens), dim=1)
        all_mask = torch.cat((audio_mask, video_mask), dim=1)
        outputs = self.transformer(inputs_embeds=all_tokens, attention_mask=all_mask)

        # Use the [CLS] token for classification
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)
        return logits


def get_dataloader(dataset: Subset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=collate,
    )


def run_epoch(
    dataloader: DataLoader,
    device: torch.device,
    model: MultimodalTransformer,
    optimizer: torch.optim.AdamW,
    criterion: nn.CrossEntropyLoss,
    is_train=True,
) -> tuple[float, float]:
    total_loss, correct, total = 0, 0, 0
    for nouns, audio, video, audio_mask, video_mask in dataloader:
        audio = audio.to(device)
        video = video.to(device)
        audio_mask = audio_mask.to(device)
        video_mask = video_mask.to(device)
        nouns = nouns.to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(audio, video, audio_mask, video_mask)
        loss = criterion(logits, nouns)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == nouns).sum().item()
        total += nouns.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(
    data_dir: str,
    checkpoint_dir: str,
    wandb_config: WandbConfiguration,
    *,
    num_epochs: int,
) -> str:
    """
    Train the multimodal transformer model.

    Args:
        data_dir (str): directory with .pt files
        checkpoint_dir (str): directory to save model checkpoints
        num_epochs (int): number of epochs to train the model for

    Returns:
        the path to the trained model file
    """
    run = wandb.init(entity=wandb_config.team, project=wandb_config.project)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VideoAudioDataset(data_dir)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split_idx = int(np.floor(0.8 * dataset_size))
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    train_dataloader = get_dataloader(train_dataset)
    val_dataloader = get_dataloader(val_dataset)

    model = MultimodalTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_accuracy = run_epoch(
            train_dataloader,
            device,
            model,
            optimizer,
            criterion,
        )

        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy = run_epoch(
                val_dataloader, device, model, optimizer, criterion, is_train=False
            )

        print(f"Epoch {epoch + 1} / {num_epochs}:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Train Accuracy: {train_accuracy:.4f}")
        print(f"   Validation Loss: {val_loss:.4f}")
        print(f"   Validation Accuracy: {val_accuracy:.4f}")
        print()
        run.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": val_loss,
                "validation_accuracy": val_accuracy,
            }
        )

        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint.pt")
        run.log_model(path=checkpoint_path, name="checkpoint")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "validation_accuracy": val_accuracy,
            },
            checkpoint_path,
        )

    trained_model_path = os.path.join(checkpoint_dir, "trained.pt")
    run.log_model(path=trained_model_path, name="trained")
    torch.save(model.state_dict(), trained_model_path)
    run.finish()
    return trained_model_path
