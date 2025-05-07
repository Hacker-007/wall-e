from typing import List, NamedTuple
import os
import numpy as np
import pandas as pd
import torch

from .find_intervals import find_annotation_intervals
from .data_fetching import download_segment
from .encoder import Encoder


class Interval(NamedTuple):
    interval_id: str
    participant_id: str
    video_id: str
    start_timestamp: str
    end_timestamp: str
    middle_noun: int


class IntervalBatch:
    df: pd.DataFrame

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_intervals(self) -> List[Interval]:
        intervals = []
        for row in self.df.itertuples(index=False):
            intervals.append(Interval(*row))

        return intervals


def get_intervals() -> pd.DataFrame:
    interval_path = "metadata/intervals.csv"
    if os.path.exists(interval_path):
        return pd.read_csv(interval_path)

    annotation_df = pd.read_csv("metadata/annotations.csv")
    intervals_df = find_annotation_intervals(annotation_df)
    sampled_df = intervals_df.groupby("middle_noun", group_keys=False).apply(
        lambda rows: rows.sample(n=1000, random_state=42)
    )

    sampled_df.to_csv(interval_path, index=False)
    return sampled_df


def batch_intervals(batch_size: int) -> List[IntervalBatch]:
    intervals_df = get_intervals()
    num_intervals = len(intervals_df)
    batches = []
    for start_row in range(0, num_intervals, batch_size):
        end_row = min(start_row + batch_size, num_intervals)
        batch_df = intervals_df.iloc[start_row:end_row]
        batch_df = batch_df.sort_values(by=["participant_id", "video_id"])
        batch_df.index = np.arange(0, len(batch_df))
        batches.append(IntervalBatch(batch_df))

    return batches


def download_interval(interval: Interval) -> str:
    file_path = f"data/{interval.interval_id}.mp4"
    download_segment(
        interval.participant_id,
        interval.video_id,
        interval.start_timestamp,
        interval.end_timestamp,
        file_path,
    )
    return file_path


def encode_video(video_path: str, interval: Interval):
    encoder = Encoder()
    tokens = encoder.process_interval(video_path)
    encoded_path = f"data/{interval.interval_id}.pt"
    torch.save(
        {
            "split_point": tokens["audio"].shape[1],
            "tokens": torch.cat([tokens["audio"], tokens["video"]], dim=1),
        },
        encoded_path,
    )

    return encoded_path


def get_tokens(interval: Interval) -> dict[str, torch.Tensor]:
    encoded_path = f"data/{interval.interval_id}.pt"
    encoded_data = torch.load(encoded_path)
    split_point = encoded_data["split_point"]
    all_tokens = encoded_data["tokens"]
    audio_tokens = all_tokens[:, :split_point]
    video_tokens = all_tokens[:, split_point:]
    return {
        "audio": audio_tokens,
        "video": video_tokens,
    }
