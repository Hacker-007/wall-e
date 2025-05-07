import os
import multiprocessing as mp

from processing import Interval, batch_intervals, download_interval, encode_video


def process_interval(interval: Interval):
    video_path = download_interval(interval)
    encoded_path = encode_video(video_path, interval)
    os.remove(video_path)
    return encoded_path


if __name__ == "__main__":
    batches = batch_intervals(1)
    batch = batches[0]
    intervals = batch.get_intervals()
    with mp.Pool(processes=5) as pool:
        pool.map(process_interval, intervals)
