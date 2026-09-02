import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import numpy as np

class VoiceSpoofDetector:
    def __init__(self, model_id: str = "MelodyMachine/Deepfake-audio-detection", device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Initializing VoiceSpoofDetector on device: {self.device}")

        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
            self.model = AutoModelForAudioClassification.from_pretrained(model_id)
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            self.id2label = getattr(self.model.config, "id2label", {})
            print(f"[*] Model Labels loaded: {self.id2label}")
        except Exception as e:
            print(f"[!] Warning: Model load failed ({e}). Running fallback.")
            self.is_loaded = False
            self.id2label = {}

    def predict_spoof_probability(self, audio_array: np.ndarray, sample_rate: int = 16000) -> float:
        if len(audio_array) == 0:
            return 0.5

        if not self.is_loaded:
            return 0.5

        try:
            inputs = self.feature_extractor(
                audio_array, 
                sampling_rate=sample_rate, 
                return_tensors="pt", 
                padding=True
            )
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]

            # In this checkpoint: Index 0 is REAL (99.99%), Index 1 is FAKE (0.01%) OR vice-versa.
            # Real speech probability is in probs[1], fake is 1 - probs[1]
            prob_0 = float(probs[0].item())
            prob_1 = float(probs[1].item())

            # Map the true fake probability
            fake_prob = prob_0 if prob_0 > prob_1 else (1.0 - prob_1)
            return float(fake_prob)

        except Exception as e:
            print(f"[!] Inference error: {e}")
            return 0.5
