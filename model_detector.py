import torch
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

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
            print("[*] Model loaded successfully.")
        except Exception as e:
            print(f"[!] Critical Model Load Failure: {e}")
            self.is_loaded = False

    def predict_spoof_probability(self, audio_array: np.ndarray, sample_rate: int = 16000) -> float:
        if len(audio_array) == 0:
            return 0.5

        if not self.is_loaded:
            print("[!] Model not loaded, returning fallback 0.5")
            return 0.5

        try:
            audio_clean = np.asarray(audio_array, dtype=np.float32).flatten()

            # Ensure minimum duration (~0.5s) to avoid shape mismatch
            min_samples = int(sample_rate * 0.5)
            if len(audio_clean) < min_samples:
                audio_clean = np.pad(audio_clean, (0, min_samples - len(audio_clean)), mode='constant')

            inputs = self.feature_extractor(
                audio_clean, 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            )
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

            # Verified mapping from actual inference logs:
            # Index 0 = Bonafide / Human (WhatsApp voice)
            # Index 1 = Deepfake / Spoof (AI cloned voices)
            fake_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])

            print(f"[INFERENCE RESULT] Probs: {probs} | True Fake Score: {fake_prob:.4f}")

            return float(np.clip(fake_prob, 0.01, 0.99))

        except Exception as e:
            import traceback
            print(f"[!] Inference exception: {e}")
            traceback.print_exc()
            return 0.5