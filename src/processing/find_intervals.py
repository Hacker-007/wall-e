from datetime import datetime, timedelta
import random
import string
import ast

import pandas as pd
from tqdm import tqdm

RELEVANT_CLASSES = [
    10,
    11,
    12,
    13,
    14,
    27,
    29,
    41,
    45,
    57,
]


def timestamp_to_seconds(t: str) -> float:
    """Convert HH:MM:SS.FF to seconds"""
    t = datetime.strptime(t, "%H:%M:%S.%f")
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6


def seconds_to_timestamp(seconds: float) -> str:
    """Convert HH:MM:SS.FF to seconds"""
    return str(timedelta(seconds=seconds))


def generate_id(length=10):
    letters = string.ascii_letters
    return "".join(random.choice(letters) for _ in range(length))


def find_annotation_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns 5-second intervals that satisfy the following constraints:
        - the verb action in the middle 1-second of the interval has no overlaps
        - there are at least two seconds of video before and after
        - there is only one verb during the middle 1-second interval
        - the verb must be one of the 10 classes that we are interested in

    Parameters:
        df (pandas.DataFrame): a DataFrame with video annotation information,
                               start and stop timestamps for each clip, and
                               the nouns responsible for the actions

    Returns:
        a DataFrame with all of the valid intervals found from the provided annotation
        DataFrame
    """
    df = df.copy()
    df["start_sec"] = df["start_timestamp"].apply(timestamp_to_seconds)
    df["stop_sec"] = df["stop_timestamp"].apply(timestamp_to_seconds)
    videos_df = pd.read_csv("metadata/video-listing.csv")
    result = []
    for video_id, group in tqdm(df.groupby("video_id")):
        min_time = group["start_sec"].min()
        max_time = group["stop_sec"].max()

        t = min_time
        while t + 3.0 <= max_time:
            window_start = t
            window_end = t + 3.0
            middle_start = t + 1.0
            middle_end = t + 2.0

            # Find all annotations that overlap with the middle 1-second interval
            overlaps = group[
                (group["start_sec"] < middle_end) & (group["stop_sec"] > middle_start)
            ]

            if len(overlaps) == 1:
                participant_id = overlaps.iloc[0]["participant_id"]
                noun_classes = ast.literal_eval(overlaps.iloc[0]["all_noun_classes"])
                has_video = (videos_df == [participant_id, video_id]).all(axis=1)
                if (
                    len(noun_classes) == 1
                    and noun_classes[0] in RELEVANT_CLASSES
                    and has_video.any()
                ):
                    interval_id = generate_id()
                    result.append(
                        {
                            "interval_id": interval_id,
                            "participant_id": participant_id,
                            "video_id": video_id,
                            "start_timestamp": seconds_to_timestamp(window_start),
                            "end_timestamp": seconds_to_timestamp(window_end),
                            "middle_noun": noun_classes[0],
                        }
                    )

            t += 1.0

    return pd.DataFrame(result)
