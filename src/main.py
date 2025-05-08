import os
from pathlib import Path
import yaml
import multiprocessing as mp
import wandb

from processing import Interval, group_intervals, download_interval, encode_video
from processing.wandb import WandbConfiguration
from train import train_model


def process_interval(interval: Interval):
    try:
        video_path = download_interval(interval)
        encoded_path = encode_video(video_path, interval)
        os.remove(video_path)
        return encoded_path
    except:
        print(f"error occurred when processing interval {interval.interval_id}")

def load_configuration(config_file: str) -> WandbConfiguration:
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    wandb_config = config['wandb']
    wandb.login(key=wandb_config['key'], verify=True)
    return WandbConfiguration(wandb_config['team'], wandb_config['project'], wandb_config['key'])


if __name__ == "__main__":
    Path("data/").mkdir(exist_ok=True)
    Path("checkpoints/").mkdir(exist_ok=True)
    group_size = 24
    groups = group_intervals(group_size)
    for group in groups:
        intervals = group.get_intervals()
        with mp.Pool(processes=group_size) as pool:
            pool.map(process_interval, intervals)

    wandb_config = load_configuration('config.yaml')
    train_model("data", "checkpoints", wandb_config, num_epochs=3)

