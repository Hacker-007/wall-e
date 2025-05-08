import os
import multiprocessing as mp

from processing import Interval, batch_intervals, download_interval, encode_video


def process_interval(interval: Interval):
    video_path = download_interval(interval)
    encoded_path = encode_video(video_path, interval)
    os.remove(video_path)
    return encoded_path


if __name__ == "__main__":
    group_size = 1
    batches = batch_intervals(group_size)
    batch = batches[0]
    intervals = batch.get_intervals()
    with mp.Pool(processes=group_size) as pool:
        pool.map(process_interval, intervals)
