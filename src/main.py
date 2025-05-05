import os

from processing import batch_intervals, download_interval, encode_video

batches = batch_intervals(1)
for batch in batches:
    interval_df = batch.iloc[0]
    video_path = download_interval(interval_df)
    encode_video(
        video_path,
        f"data/{interval_df['interval_id']}.pt",
    )

    os.remove(video_path)