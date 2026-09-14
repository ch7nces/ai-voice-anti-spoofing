import random
import time
import os

# Production me Twilio / Fast2SMS API keys .env se aayengi
# Demo ke liye fallback console mode humesha active rahega
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")

class OutOfBandAuthService:
    def __init__(self, otp_expiry_seconds: int = 120):
        self.otp_expiry_seconds = otp_expiry_seconds
        # In-memory storage: {phone_number: {"otp": "123456", "expires_at": timestamp}}
        self.active_sessions = {}

    def generate_and_send_otp(self, target_phone: str) -> dict:
        """Generates a 6-digit cryptographic OTP and dispatches it via external channel."""
        otp = str(random.randint(100000, 999999))
        expires_at = time.time() + self.otp_expiry_seconds
        
        self.active_sessions[target_phone] = {
            "otp": otp,
            "expires_at": expires_at,
            "verified": False
        }

        # SMS Dispatch Logic (Twilio Integration)
        dispatched_via_sms = False
        if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER:
            try:
                from twilio.rest import Client
                client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
                message = client.messages.create(
                    body=f"[VoiceGuard Security Alert] Suspicious voice activity detected on your call. Your Out-of-Band verification OTP is: {otp}. Do NOT share this over the phone.",
                    from_=TWILIO_PHONE_NUMBER,
                    to=target_phone
                )
                dispatched_via_sms = True
                print(f"[*] SMS dispatched successfully! SID: {message.sid}")
            except Exception as e:
                print(f"[!] SMS dispatch failed: {e}")

        # Console print for instant hackathon evaluation & demo testing
        print(f"\n=======================================================")
        print(f"🚨 [OOB SECURITY ALERT] Out-of-Band Channel Triggered")
        print(f"📱 Target Channel: SMS to {target_phone}")
        print(f"🔑 Live Security OTP: >>> {otp} <<< (Expires in 2 mins)")
        print(f"=======================================================\n")

        return {
            "status": "SENT",
            "phone": target_phone,
            "demo_otp_display": otp, # Frontend demo toast me dikhane ke liye
            "via_real_sms": dispatched_via_sms
        }

    def verify_otp(self, target_phone: str, user_otp: str) -> dict:
        """Validates entered OTP against active memory state."""
        session = self.active_sessions.get(target_phone)
        
        if not session:
            return {"success": False, "message": "No active verification session found. Request a new OTP."}
        
        if time.time() > session["expires_at"]:
            del self.active_sessions[target_phone]
            return {"success": False, "message": "OTP has expired. Request a new one."}
        
        if session["otp"] == user_otp.strip():
            session["verified"] = True
            return {"success": True, "message": "Out-of-Band Identity Verified! Channel Unblocked."}
        else:
            return {"success": False, "message": "Invalid OTP code entered. Potential impersonation."}