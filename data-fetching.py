import os
import subprocess

DATASET_BASE_URL = "https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m"


def download_segment(
    participant_id: str,
    video_id: str,
    start_timestamp: str,
    stop_timestamp: str,
    output_path: str,
):
    """
    Downloads a clip from a remote data set server using ffmpeg without re-encoding.

    Parameters:
        participant_id (str): ID of the participant who created the video
        video_id (str): the video ID of the participant to download
        start_timestamp (str): Start timestamp in HH:MM:SS.ss format
        stop_timestamp (str): End timestamp in HH:MM:SS.ss format
        output_path (str): Path to save the clip (e.g., "my_clip.mp4").

    Returns:
        bool: True if successful, False otherwise.
    """

    url = f"{DATASET_BASE_URL}/{participant_id}/videos/{video_id}.MP4"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    command = [
        "ffmpeg",
        "-ss",
        start_timestamp,
        "-to",
        stop_timestamp,
        "-i",
        url,
        "-c",
        "copy",
        "-y",
        output_path,
    ]

    try:
        subprocess.run(command, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(
            "[decode error] error occurred when decoding the remote file:",
            e.stderr.decode(),
        )
        return False


download_segment("P01", "P01_104", "00:00:05", "00:00:10", "test.mp4")
