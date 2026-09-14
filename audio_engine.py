import numpy as np
import librosa
import soundfile as sf
import io
import tempfile
import os
from pydub import AudioSegment

class AudioFeatureExtractor:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def load_audio_bytes(self, audio_bytes: bytes) -> np.ndarray:
        # Strategy 1: soundfile
        try:
            with io.BytesIO(audio_bytes) as audio_file:
                y, sr = sf.read(audio_file)
                if len(y.shape) > 1:
                    y = np.mean(y, axis=1)
                if sr != self.sample_rate:
                    y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=self.sample_rate)
                y = self._trim_silence(y)
                return y.astype(np.float32)
        except Exception:
            pass

        # Strategy 2: pydub
        try:
            audio_seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
            audio_seg = audio_seg.set_channels(1).set_frame_rate(self.sample_rate)
            samples = np.array(audio_seg.get_array_of_samples(), dtype=np.float32)
            max_val = float(2 ** (audio_seg.sample_width * 8 - 1))
            y = samples / max_val
            y = self._trim_silence(y)
            return y.astype(np.float32)
        except Exception:
            pass

        # Strategy 3: tempfile fallback
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_path = temp_audio.name

        try:
            y, _ = librosa.load(temp_path, sr=self.sample_rate, mono=True)
            y = self._trim_silence(y)
            return y.astype(np.float32)
        except Exception as e:
            print(f"[!] Audio decode error: {e}")
            return np.zeros(self.sample_rate * 2, dtype=np.float32)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _trim_silence(self, y: np.ndarray) -> np.ndarray:
        """Removes leading and trailing dead air so short speech isn't skewed."""
        if len(y) == 0:
            return y
        trimmed, _ = librosa.effects.trim(y, top_db=25)
        return trimmed if len(trimmed) > 0 else y

    def extract_prosodic_features(self, y: np.ndarray) -> dict:
        duration = len(y) / float(self.sample_rate)
        if duration < 0.25:
            return {
                "pitch_variance": 0.0,
                "spectral_flatness": 0.0,
                "spectral_rolloff": 0.0,
                "energy_entropy": 0.0,
                "is_silent": True
            }

        # Pitch variation (F0)
        try:
            f0, voiced_flag, _ = librosa.pyin(
                y,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate
            )
            valid_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
            pitch_variance = float(np.var(valid_f0)) if len(valid_f0) > 1 else 0.0
        except Exception:
            pitch_variance = 0.0

        # Spectral dynamics
        try:
            flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
            rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=self.sample_rate, roll_percent=0.85)))
        except Exception:
            flatness = 0.0
            rolloff = 0.0

        # Energy entropy check: Real human voice bursts have high localized frame variance
        try:
            rms = librosa.feature.rms(y=y)[0]
            rms_var = float(np.var(rms))
        except Exception:
            rms_var = 0.0

        return {
            "pitch_variance": pitch_variance,
            "spectral_flatness": flatness,
            "spectral_rolloff": rolloff,
            "rms_variance": rms_var,
            "duration": duration,
            "is_silent": False
        }

    def compute_mel_spectrogram(self, y: np.ndarray, n_mels: int = 128) -> np.ndarray:
        if len(y) == 0:
            return np.zeros((n_mels, 10))
        mel_spec = librosa.feature.melspectrogram(y=y, sr=self.sample_rate, n_mels=n_mels)
        return librosa.power_to_db(mel_spec, ref=np.max)