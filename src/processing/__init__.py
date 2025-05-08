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


class EncodedData(NamedTuple):
    noun: str
    audio: torch.Tensor
    video: torch.Tensor


class IntervalGroup:
    df: pd.DataFrame

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_intervals(self) -> List[Interval]:
        intervals = []
        for row in self.df.itertuples(index=False):
            intervals.append(Interval(*row))

        return intervals


def get_intervals() -> pd.DataFrame:
    interval_path = os.path.join("metadata", "intervals.csv")
    if os.path.exists(interval_path):
        return pd.read_csv(interval_path)

    annotation_df = pd.read_csv(os.path.join("metadata", "annotations.csv"))
    intervals_df = find_annotation_intervals(annotation_df)

    # Randomly sample 300 videos per noun to create our "dataset"
    sampled_df = intervals_df.groupby("middle_noun", group_keys=False).apply(
        lambda rows: rows.sample(n=300, random_state=42)
    )

    sampled_df.to_csv(interval_path, index=False)
    return sampled_df


def group_intervals(group_size: int, is_train=True) -> List[IntervalGroup]:
    intervals_df = get_intervals()
    groups = []
    if is_train:
        interval_start, interval_end = 0, 2 * (len(intervals_df) // 3)
    else:
        interval_start, interval_end = 2 * (len(intervals_df) // 3), len(intervals_df)

    for start_row in range(interval_start, interval_end, group_size):
        end_row = min(start_row + group_size, interval_end)
        group_df = intervals_df.iloc[start_row:end_row]
        group_df = group_df.sort_values(by=["participant_id", "video_id"])
        group_df.index = np.arange(0, len(group_df))
        groups.append(IntervalGroup(group_df))

    return groups


def download_interval(interval: Interval) -> str:
    file_path = os.path.join("data", f"{interval.interval_id}.mp4")
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
    encoded_path = os.path.join("data", f"{interval.interval_id}.pt")
    torch.save(
        {
            "noun": interval.middle_noun,
            "split_point": tokens["audio"].shape[1],
            "tokens": torch.cat([tokens["audio"], tokens["video"]], dim=1),
        },
        encoded_path,
    )

    return encoded_path


def get_tokens(encoded_path: str) -> EncodedData:
    encoded_data = torch.load(encoded_path)
    split_point = encoded_data["split_point"]
    all_tokens = encoded_data["tokens"]
    audio_tokens = all_tokens[:, :split_point]
    video_tokens = all_tokens[:, split_point:]
    return EncodedData(
        encoded_data["noun"],
        audio_tokens,
        video_tokens,
    )
