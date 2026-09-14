import torch
import soundfile as sf
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

model_id = "ahmedyousri/distilhubert-base-anti-spoofing"
print("Loading model...")
extractor = AutoFeatureExtractor.from_pretrained(model_id)
model = AutoModelForAudioClassification.from_pretrained(model_id)

# 1-second dummy 16kHz audio
dummy_audio = np.random.uniform(-0.1, 0.1, 16000).astype(np.float32)

inputs = extractor(dummy_audio, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].numpy()

print(f"Success! Probs: {probs}")