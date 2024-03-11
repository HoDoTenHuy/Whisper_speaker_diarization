import os
import time
import shutil
import subprocess

from io import BytesIO
from pathlib import Path
from fastapi import UploadFile
from tempfile import NamedTemporaryFile


def time_fn(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print(f"Elapsed time: {(end - start)}ms")

    return wrapper


def convert_mp42wav(video_file):
    audio_file = video_file.replace('mp4', 'wav')
    command = [
        "ffmpeg",
        "-i", video_file,
        "-ar", "16000",  # Audio sample rate (Hz)
        "-ac", "1",  # Number of audio channels (1 for mono)
        "-c:a", "pcm_s16le",  # Audio codec (PCM signed 16-bit little-endian)
        audio_file
    ]
    subprocess.run(command)


def convert_mp32wav(mp3_file):
    audio_file = mp3_file.replace('mp3', 'wav')
    command = [
        "ffmpeg",
        "-i", mp3_file,
        "-ar", "44100",
        "-ac", "2",
        "-c:a", "pcm_s16le",
        audio_file
    ]
    subprocess.run(command)


def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    try:
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = Path(tmp.name)
    finally:
        upload_file.file.close()
    return tmp_path


def delete_tmp_file(path: str):
    if os.path.exists(path):
        os.remove(path)


if __name__ == '__main__':
    convert_mp32wav('../audio/video_3.mp4')
