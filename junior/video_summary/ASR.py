import torch
import whisper


def transcribe(file_path: str, model_name="base") -> str:
    """
    Transcribe input audio file.

    Examples
    --------
    # >>> text = transcribe(".../audio.mp3")
    # >>> print(text)
    'This text explains...'
    """
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model = whisper.load_model(model_name, device=device)
    result = model.transcribe(file_path, device=device)
    print(result["text"])
    return result["text"]


transcribe(r"C:\Users\Neesty\PycharmProjects\ml_sim_kc\junior\video_summary\audio.mp3")