from collections import deque
import numpy as np

class DynamicRiskScorer:
    def __init__(self, window_size: int = 3, soft_threshold: float = 45.0, hard_threshold: float = 75.0):
        self.window_size = window_size
        self.soft_threshold = soft_threshold
        self.hard_threshold = hard_threshold
        self.score_history = deque(maxlen=window_size)

    def calculate_risk(self, spoof_prob: float, prosody_features: dict) -> dict:
        pitch_var = prosody_features.get("pitch_variance", 0.0)
        flatness = prosody_features.get("spectral_flatness", 0.0)
        is_silent = prosody_features.get("is_silent", False)

        if is_silent:
            return {
                "instant_score": 0.0,
                "smoothed_risk_score": 0.0,
                "status": "INSUFFICIENT_AUDIO",
                "message": "Audio silent or too short.",
                "badge_color": "gray",
                "window_samples": 0
            }

        # PRIORITY 1: Extreme Vocoder Anomalies (High Pitch Glitch > 1500 OR Synthetic Flatness > 0.25)
        if pitch_var > 1500.0 or flatness > 0.25:
            instant_risk = 92.0

        # PRIORITY 2: Proven Biological Vocal Cord Intonation
        # Real humans naturally vary pitch between 30 and 1000 with organic harmonics (flatness <= 0.18)
        elif 30.0 <= pitch_var <= 1000.0 and flatness <= 0.18:
            instant_risk = 8.0

        # PRIORITY 3: Live mic recording with ambient background noise
        elif 18.0 <= pitch_var <= 1400.0 and flatness <= 0.23:
            instant_risk = 15.0

        # PRIORITY 4: Robotic Monotone / Flat Pitch (< 15) with zero intonation
        else:
            instant_risk = 85.0

        # Update sliding buffer
        self.score_history.append(instant_risk)
        rolling_risk = float(np.mean(self.score_history))

        if rolling_risk >= self.hard_threshold:
            status = "CRITICAL_ALERT"
            message = "High-confidence synthetic voice detected. Block transaction/escalate."
            badge_color = "red"
        elif rolling_risk >= self.soft_threshold:
            status = "SUSPICIOUS"
            message = "Unusual acoustic signatures. Multi-factor verification suggested."
            badge_color = "orange"
        else:
            status = "VERIFIED_HUMAN"
            message = "Natural human voice parameters verified."
            badge_color = "green"

        return {
            "instant_score": round(instant_risk, 2),
            "smoothed_risk_score": round(rolling_risk, 2),
            "status": status,
            "message": message,
            "badge_color": badge_color,
            "window_samples": len(self.score_history)
        }

    def reset(self):
        self.score_history.clear()
