import pandas as pd
from datetime import datetime, timedelta
import ast


def timestamp_to_seconds(t: str) -> float:
    """Convert HH:MM:SS.FF to seconds"""
    t = datetime.strptime(t, "%H:%M:%S.%f")
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6


def seconds_to_timestamp(seconds: float) -> str:
    """Convert HH:MM:SS.FF to seconds"""
    return str(timedelta(seconds=seconds))


def find_annotation_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns 5-second intervals that satisfy the following constraints:
        - the verb action in the middle 1-second of the interval has no overlaps
        - there are at least two seconds of video before and after
        - there is only one verb during the middle 1-second interval

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
    result = []
    for video_id, group in df.groupby("video_id"):
        min_time = group["start_sec"].min()
        max_time = group["stop_sec"].max()

        t = min_time
        while t + 5.0 <= max_time:
            window_start = t
            window_end = t + 5.0
            middle_start = t + 2.0
            middle_end = t + 3.0

            # Find all annotations that overlap with the middle 1-second interval
            overlaps = group[
                (group["start_sec"] < middle_end) & (group["stop_sec"] > middle_start)
            ]

            if len(overlaps) == 1:
                all_noun_classes = overlaps.iloc[0]["all_noun_classes"]
                narration_id = overlaps.iloc[0]["narration_id"]
                participant_id = overlaps.iloc[0]["participant_id"]
                noun_classes = ast.literal_eval(all_noun_classes)
                if len(noun_classes) == 1:
                    result.append(
                        {
                            "narration_id": narration_id,
                            "participant_id": participant_id,
                            "video_id": video_id,
                            "start_timestamp": seconds_to_timestamp(
                                round(window_start, 2)
                            ),
                            "end_timestamp": seconds_to_timestamp(round(window_end, 2)),
                            "middle_noun": noun_classes[0],
                        }
                    )

            t += 1.0

    return pd.DataFrame(result)
