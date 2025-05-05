import os
import subprocess
import requests

DATASET_URL_FORMATS = [
    "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/<participant_id>/<video_id>.MP4",
    "https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m/<participant_id>/videos/<video_id>.MP4",
]

def get_data_url(participant_id: str, video_id: str) -> str:
    for url in DATASET_URL_FORMATS:
        url = url.replace("<participant_id>", participant_id)
        url = url.replace("<video_id>", video_id)
        response = requests.head(url)
        if response.status_code == 200:
            return url
        
    raise Exception(f'{participant_id}/{video_id} does not exist')

def download_segment(
    participant_id: str,
    video_id: str,
    start_timestamp: str,
    stop_timestamp: str,
    output_path: str,
) -> bool:
    """
    Downloads a clip from a remote data set server using ffmpeg without re-encoding.

    Parameters:
        participant_id (str): ID of the participant who created the video
        video_id (str): the video ID of the participant to download
        start_timestamp (str): start timestamp in HH:MM:SS.ss format
        stop_timestamp (str): end timestamp in HH:MM:SS.ss format
        output_path (str): path to save the clip (e.g., "my_clip.mp4")

    Returns:
        bool: True if successful, False otherwise.
    """

    url = get_data_url(participant_id, video_id)
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
        "-preset",
        "ultrafast",
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
