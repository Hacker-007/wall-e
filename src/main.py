import os
from pathlib import Path
import multiprocessing as mp

from processing import Interval, batch_intervals, download_interval, encode_video
from train import train_model


def process_interval(interval: Interval):
    try:
        video_path = download_interval(interval)
        encoded_path = encode_video(video_path, interval)
        os.remove(video_path)
        return encoded_path
    except:
        print(f"error occurred when processing interval {interval.interval_id}")


if __name__ == "__main__":
    Path('data/').mkdir(exist_ok=True)
    Path('checkpoints/').mkdir(exist_ok=True)
    group_size = 5
    batches = batch_intervals(group_size)
    batch = batches[0]
    intervals = batch.get_intervals()
    with mp.Pool(processes=group_size) as pool:
        pool.map(process_interval, intervals)
    
    train_model("data", "checkpoints")