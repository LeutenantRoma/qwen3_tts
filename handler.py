import base64
import io
import os
import torch
import runpod
import soundfile as sf

from transformers import AutoProcessor, AutoModel

MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEVICE = "cuda"

# Optional: cache models (important for RunPod)
os.environ["HF_HOME"] = "/runpod-volume/hf-cache"

print("Loading model...")

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    trust_remote_code=True
).to(DEVICE)

model.eval()

print("Model loaded.")

def handler(job):
    text = job["input"].get("text", "Hello world")
    speaker = job["input"].get("speaker", "default")
    language = job["input"].get("language", "en")

    inputs = processor(
        text=text,
        speaker=speaker,
        language=language,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        audio = model.generate(**inputs)

    buffer = io.BytesIO()
    sf.write(buffer, audio.cpu().numpy(), samplerate=22050, format="WAV")

    encoded = base64.b64encode(buffer.getvalue()).decode()

    return {
        "audio_base64": encoded
    }

runpod.serverless.start({"handler": handler})