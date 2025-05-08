import os
from pathlib import Path
from tqdm import tqdm
import yaml
import wandb
from concurrent.futures import ThreadPoolExecutor

from processing import Interval, group_intervals, download_interval, encode_video
from processing.encoder import Encoder
from processing.wandb import WandbConfiguration
from train import train_model


def process_interval(args: tuple[Encoder, Interval]):
    try:
        encoder, interval = args
        video_path = download_interval(interval)
        encode_video(encoder, video_path, interval)
        os.remove(video_path)
    except Exception as e:
        print(f"Error occurred when processing interval {interval.interval_id}: {e}")


def load_configuration(config_file: str) -> WandbConfiguration:
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    wandb_config = config["wandb"]
    wandb.login(key=wandb_config["key"], verify=True)
    return WandbConfiguration(
        wandb_config["team"],
        wandb_config["project"],
        wandb_config["key"],
    )


if __name__ == "__main__":
    Path("data/").mkdir(exist_ok=True)
    Path("checkpoints/").mkdir(exist_ok=True)
    group_size = 10
    groups = group_intervals(group_size)
    encoder = Encoder()
    for group in tqdm(groups):
        intervals = group.get_intervals()
        args_list = [(encoder, interval) for interval in intervals]
        with ThreadPoolExecutor(max_workers=group_size) as executor:
            for args in args_list:
                executor.submit(process_interval, args)

    wandb_config = load_configuration("config.yaml")
    train_model("data", "checkpoints", wandb_config, num_epochs=3)
