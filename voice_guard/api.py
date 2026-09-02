from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import uvicorn
import asyncio

from audio_engine import AudioFeatureExtractor
from model_detector import VoiceSpoofDetector
from risk_scorer import DynamicRiskScorer

app = FastAPI(title="VoiceGuard Anti-Spoofing Engine", version="1.0.0")

# Allow CORS for external web dashboards/telecom clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize singletons for low latency
audio_extractor = AudioFeatureExtractor()
detector = VoiceSpoofDetector()

@app.get("/")
def health_check():
    return {"status": "active", "service": "VoiceGuard Detection API"}

@app.post("/analyze-file")
async def analyze_audio_file(file: UploadFile = File(...)):
    """
    Analyzes an uploaded .wav / .mp3 audio file end-to-end.
    """
    file_bytes = await file.read()
    y = audio_extractor.load_audio_bytes(file_bytes)
    
    prosody = audio_extractor.extract_prosodic_features(y)
    spoof_prob = detector.predict_spoof_probability(y)
    
    scorer = DynamicRiskScorer()
    risk_result = scorer.calculate_risk(spoof_prob, prosody)
    
    return {
        "filename": file.filename,
        "duration_seconds": round(len(y) / 16000.0, 2),
        "risk_assessment": risk_result,
        "prosody_metrics": prosody
    }

@app.websocket("/ws/live-stream")
async def live_audio_stream(websocket: WebSocket):
    """
    Websocket endpoint for continuous live-call audio chunk processing.
    Expects raw PCM/wav byte chunks from caller/VoIP bridge.
    """
    await websocket.accept()
    session_scorer = DynamicRiskScorer(window_size=5)
    print("[*] Client connected to live-stream channel.")

    try:
        while True:
            # Receive audio chunk bytes from caller/dashboard
            chunk_bytes = await websocket.receive_bytes()
            if not chunk_bytes:
                continue

            y_chunk = audio_extractor.load_audio_bytes(chunk_bytes)
            prosody = audio_extractor.extract_prosodic_features(y_chunk)
            spoof_prob = detector.predict_spoof_probability(y_chunk)
            
            risk_result = session_scorer.calculate_risk(spoof_prob, prosody)

            # Send back real-time risk assessment immediately
            await websocket.send_json({
                "type": "RISK_UPDATE",
                "data": risk_result
            })

    except WebSocketDisconnect:
        print("[*] Client disconnected from live-stream channel.")
    except Exception as e:
        print(f"[!] WebSocket error: {e}")
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
