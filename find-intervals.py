import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import ast


def timestamp_to_seconds(t: str):
    """Convert HH:MM:SS.FF to seconds"""
    t = datetime.strptime(t, "%H:%M:%S.%f")
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6


def seconds_to_timestamp(seconds: float):
    """Convert HH:MM:SS.FF to seconds"""
    return str(timedelta(seconds=seconds))


def find_free_1s_intervals(df, step=1.0):
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
                all_nouns = overlaps.iloc[0]["all_nouns"]
                narration_id = overlaps.iloc[0]["narration_id"]
                participant_id = overlaps.iloc[0]["participant_id"]
                nouns = ast.literal_eval(all_nouns)
                if len(nouns) == 1:
                    result.append(
                        {
                            "narration_id": narration_id,
                            "participant_id": participant_id,
                            "video_id": video_id,
                            "start_timestamp": seconds_to_timestamp(round(window_start, 2)),
                            "end_timestamp": seconds_to_timestamp(round(window_end, 2)),
                            "all_nouns": all_nouns,
                        }
                    )

            t += step

    return pd.DataFrame(result)


if __name__ == "__main__":
    # annotations = pd.read_csv("metadata/annotations.csv")
    # intervals = find_free_1s_intervals(annotations)
    # intervals.to_csv('metadata/intervals.csv', index=False)
    
    intervals = pd.read_csv("metadata/intervals.csv")
    all_nouns = intervals['all_nouns']
    noun_counts = defaultdict(int)
    for l in all_nouns:
        l = ast.literal_eval(l)
        for noun in l:
            noun_counts[noun] += 1
