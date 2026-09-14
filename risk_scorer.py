from collections import deque
import numpy as np

class DynamicRiskScorer:
    def __init__(self, window_size: int = 3, soft_threshold: float = 45.0, hard_threshold: float = 70.0):
        self.window_size = window_size
        self.soft_threshold = soft_threshold
        self.hard_threshold = hard_threshold
        self.score_history = deque(maxlen=window_size)

    def calculate_risk(self, spoof_prob: float, prosody_features: dict) -> dict:
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

        # Direct linear mapping from spoof probability
        instant_risk = float(spoof_prob * 100.0)

        # Update sliding history
        self.score_history.append(instant_risk)
        rolling_risk = float(np.mean(self.score_history))

        if rolling_risk >= self.hard_threshold:
            status = "CRITICAL_ALERT"
            message = "Synthetic / AI-Cloned Audio Signature Detected!"
            badge_color = "red"
        elif rolling_risk >= self.soft_threshold:
            status = "SUSPICIOUS"
            message = "Ambiguous acoustic signals. Verification required."
            badge_color = "orange"
        else:
            status = "VERIFIED_HUMAN"
            message = "Natural biological human voice verified."
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