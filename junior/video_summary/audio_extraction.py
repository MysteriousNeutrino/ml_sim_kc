import os

from moviepy.editor import AudioFileClip
from pytube import YouTube


def video_title(youtube_url: str) -> str:
    """
    Retrieve the title of a YouTube video.

    Examples
    --------
    # >>> title = video_title("https://www.youtube.com/watch?v=svK8fFDMgzA")
    # >>> print(title)
    'Sample Video Title'
    """
    yt = YouTube(youtube_url)
    return yt.title


def download_audio(youtube_url: str, download_path: str) -> None:
    """

    :param youtube_url:
    :param download_path:
    :return:
    """
    directory = os.path.dirname(download_path)
    filename = os.path.basename(download_path)

    try:
        yt = YouTube(youtube_url)

        audio_streams = yt.streams.filter(only_audio=True)

        audio_stream = audio_streams.first()

        audio_file = audio_stream.download(output_path=directory, filename=filename)
        if os.path.getsize(audio_file) < 1000:
            raise Exception("Downloaded file is too small")

        print("Audio downloaded successfully.")
    except Exception as e:
        print(f"Error: {e}")
        raise


def convert_mp4_to_mp3(input_path: str, output_path: str) -> None:
    """
    Convert an audio file from mp4 format to mp3.

    Examples
    --------
    # >>> convert_mp4_to_mp3("path/to/audio.mp4", "path/to/audio.mp3")
    """

    audioclip = AudioFileClip(input_path)
    audioclip.write_audiofile(output_path, bitrate='320k', codec='mp3')
