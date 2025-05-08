import os
import warnings
from io import BytesIO

import cv2
import numpy as np
import torch
from pydub import AudioSegment
from transformers import AutoModel, AutoImageProcessor, AutoProcessor

# Since we are only encoding the input videos,
# we can ignore the following warnings that pertain to
# training the models
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="transformers.configuration_utils",
    message=".*gradient_checkpointing.*",
)


class Encoder:
    def __init__(
        self,
        audio_model_name="facebook/wav2vec2-base",
        video_model_name="google/vit-base-patch16-224",
    ):
        """
        Initialize the Encoder with pre-trained audio and video models.

        Args:
            audio_model_name (str): Name of the pre-trained audio model.
            video_model_name (str): Name of the pre-trained video model.
        """
        # Load pre-trained audio model and processor by setting
        # the models to evaluation mode
        self.audio_processor = AutoProcessor.from_pretrained(audio_model_name)
        self.audio_model = AutoModel.from_pretrained(audio_model_name)
        self.audio_model.eval()

        self.video_processor = AutoImageProcessor.from_pretrained(
            video_model_name,
            use_fast=True,
        )
        self.video_model = AutoModel.from_pretrained(
            video_model_name,
            add_pooling_layer=False,
        )
        self.video_model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.audio_processor = self.audio_processor.to(self.device)
        self.audio_model = self.audio_model.to(self.device)
        self.video_processor = self.video_processor.to(self.device)
        self.video_model = self.video_model.to(self.device)

        self.sample_rate = 16000
        self.fps = 25
        self.frame_size = (224, 224)

    def process_interval(self, file_path: str) -> dict[str, torch.Tensor]:
        """
        Process an MP4 file and return audio and video tokens.

        Args:
            file_path (str): Path to the MP4 file.

        Returns:
            dict: Dictionary containing audio and video tokens.
        """
        audio, frames = self._load_mp4(file_path)
        audio_tokens = self._encode_audio(audio)
        video_tokens = self._encode_video(frames)

        assert audio_tokens.shape[2] == 768
        assert video_tokens.shape[2] == 768
        return {
            "audio": audio_tokens,
            "video": video_tokens,
        }

    def _load_mp4(self, file_path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Load and validate the MP4 file, extracting audio with pydub and video with OpenCV.

        Args:
            file_path (str): Path to the MP4 file.

        Returns:
            tuple: (audio, video_frames)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")

        # Load audio using pydub
        audio_segment = AudioSegment.from_file(file_path, format="mp4")
        audio_segment = audio_segment.set_channels(1).set_frame_rate(self.sample_rate)

        buffer = BytesIO()
        audio_segment.export(buffer, format="wav")
        audio_segment = AudioSegment.from_file(buffer, format="wav")
        # Normalize to [-1, 1]
        audio = (
            np.array(audio_segment.get_array_of_samples(), dtype=np.float32) / 32768.0
        )

        cap = cv2.VideoCapture(file_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, self.frame_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()
        duration = len(audio) / self.sample_rate
        if abs(duration - 3.0) > 0.1:
            raise ValueError(f"MP4 duration is {duration:.2f}s, expected ~3s.")

        return audio, np.array(frames)

    def _encode_audio(self, audio: np.ndarray) -> torch.Tensor:
        """
        Encode audio into tokens using the pre-trained audio model.

        Args:
            audio (np.ndarray): Raw audio waveform.

        Returns:
            torch.Tensor: Audio tokens.
        """
        inputs = self.audio_processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.audio_model(**inputs)

        # Extract hidden states as tokens (e.g., last hidden state)
        # shape: (1, sequence length, hidden_size)
        return outputs.last_hidden_state

    def _encode_video(self, frames: np.ndarray) -> torch.Tensor:
        """
        Encode video frames into tokens using the pre-trained video model. The returned
        video tokens only include the CLS embedding for each frame to maintain a
        reasonable number of tokens in the total model.

        Args:
            frames (np.ndarray): Array of video frames (num_frames, height, width, channels).

        Returns:
            torch.Tensor: Video tokens.
        """
        total_frames = len(frames)
        target_fps = int(self.fps)
        total_duration = 3.0

        # Calculate frame indices for first second (0-1s) and last second (2-3s)
        fps = total_frames / total_duration
        first_sec_frames = int(fps * 1.0)
        last_sec_start = int(fps * 2.0)
        last_sec_frames = total_frames - last_sec_start

        # Subsample to match target FPS (25 frames per second)
        first_sec_indices = np.linspace(
            0, first_sec_frames - 1, min(first_sec_frames, target_fps)
        ).astype(int)
        last_sec_indices = np.linspace(
            last_sec_start, total_frames - 1, min(last_sec_frames, target_fps)
        ).astype(int)

        # Select frames
        selected_indices = np.concatenate([first_sec_indices, last_sec_indices])
        selected_frames = frames[selected_indices]

        # Preprocess frames
        inputs = self.video_processor(images=list(selected_frames), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Encode frames with frozen model
        with torch.no_grad():
            outputs = self.video_model(**inputs)

        # Extract only CLS embeddings as tokens
        # shape: (1, num_frames, hidden_size)
        video_tokens = outputs.last_hidden_state
        return video_tokens[:, 0, :].unsqueeze(0)
