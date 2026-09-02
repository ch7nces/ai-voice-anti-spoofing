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
        """
        Safely converts raw audio bytes from any source (MP3, WAV, WebM, OGG, M4A)
        into a 16kHz mono float32 numpy array.
        """
        # Strategy 1: Standard soundfile read
        try:
            with io.BytesIO(audio_bytes) as audio_file:
                y, sr = sf.read(audio_file)
                if len(y.shape) > 1:
                    y = np.mean(y, axis=1)
                if sr != self.sample_rate:
                    y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=self.sample_rate)
                return y.astype(np.float32)
        except Exception:
            pass

        # Strategy 2: Decode via Pydub into raw PCM
        try:
            audio_seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
            audio_seg = audio_seg.set_channels(1).set_frame_rate(self.sample_rate)
            samples = np.array(audio_seg.get_array_of_samples(), dtype=np.float32)
            # Normalize to [-1.0, 1.0]
            max_val = float(2 ** (audio_seg.sample_width * 8 - 1))
            y = samples / max_val
            return y.astype(np.float32)
        except Exception:
            pass

        # Strategy 3: Fallback temporary file with librosa
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_path = temp_audio.name

        try:
            y, _ = librosa.load(temp_path, sr=self.sample_rate, mono=True)
            return y.astype(np.float32)
        except Exception as e:
            print(f"[!] Critical audio decoding failure: {e}")
            return np.zeros(self.sample_rate * 2, dtype=np.float32)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def extract_prosodic_features(self, y: np.ndarray) -> dict:
        """Extracts pitch variations (F0), spectral flatness and centroid."""
        if len(y) < self.sample_rate * 0.3:
            return {"pitch_variance": 0.0, "spectral_flatness": 0.0, "is_silent": True}

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

        try:
            flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
            centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=self.sample_rate)))
        except Exception:
            flatness = 0.0
            centroid = 0.0

        return {
            "pitch_variance": pitch_variance,
            "spectral_flatness": flatness,
            "spectral_centroid": centroid,
            "is_silent": False
        }

    def compute_mel_spectrogram(self, y: np.ndarray, n_mels: int = 128) -> np.ndarray:
        """Extracts Log-Mel Spectrogram for visual inspection."""
        if len(y) == 0:
            return np.zeros((n_mels, 10))
        mel_spec = librosa.feature.melspectrogram(y=y, sr=self.sample_rate, n_mels=n_mels)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec
