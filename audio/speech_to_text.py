import os
os.environ["HF_HOME"] = "D:\\hf-cache"
from transformers import pipeline

asr = pipeline(
    "automatic-speech-recognition",
    model = "openai/whisper-small"
)

audio_path = r"D:\ml-applications\audio\test.mp3"

res = asr(
    audio_path,
    generate_kwargs = {"language": "russian", "task": "transcribe"}
)
print(res["text"])