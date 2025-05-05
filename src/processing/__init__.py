from typing import List
import os
import pandas as pd
import torch

from .find_intervals import find_annotation_intervals
from .data_fetching import download_segment
from .encoder import Encoder


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


def batch_intervals(num_groups: int) -> List[pd.DataFrame]:
    intervals_df = get_intervals()
    num_intervals = len(intervals_df)
    batches = []
    batch_size = num_intervals // num_groups
    for start_row in range(0, num_intervals, batch_size):
        end_row = min(start_row + batch_size, num_intervals)
        batch_df = intervals_df.iloc[start_row:end_row]
        batch_df = batch_df.sort_values(by=["participant_id", "video_id"])
        batches.append(batch_df)

    return batches


def download_interval(interval_df: pd.Series) -> str:
    [
        interval_id,
        participant_id,
        video_id,
        start,
        end,
        _,
    ] = interval_df
    download_segment(participant_id, video_id, start, end, f"data/{interval_id}.mp4")
    return f"data/{interval_id}.mp4"


def encode_video(video_path: str, encoding_path: str):
    encoder = Encoder()
    tokens = encoder.process_interval(video_path)
    all_tokens = torch.concatenate([tokens["audio"], tokens["video"]], dim=1)
    torch.save(all_tokens, encoding_path)
