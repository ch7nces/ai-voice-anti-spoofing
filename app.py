import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io
import soundfile as sf

from audio_engine import AudioFeatureExtractor
from model_detector import VoiceSpoofDetector
from risk_scorer import DynamicRiskScorer
from otp_service import OutOfBandAuthService

st.set_page_config(
    page_title="VoiceGuard | AI Voice Cloning Detection",
    page_icon="🛡️",
    layout="wide"
)

# Initialize components
if "extractor" not in st.session_state:
    st.session_state.extractor = AudioFeatureExtractor()
    st.session_state.detector = VoiceSpoofDetector()
    st.session_state.scorer = DynamicRiskScorer()
    st.session_state.otp_service = OutOfBandAuthService()
    st.session_state.otp_sent = False
    st.session_state.registered_phone = "+919876543210"

extractor = st.session_state.extractor
detector = st.session_state.detector
scorer = st.session_state.scorer

# UI Header
st.title("🛡️ VoiceGuard: Real-Time AI Voice Impersonation Defense")
st.markdown("AI-driven acoustic and prosodic verification framework to detect neural voice cloning in VoIP, telebanking, and high-risk workflows.")
st.divider()

# Sidebar Settings
st.sidebar.header("⚙️ Detection Parameters")
soft_thresh = st.sidebar.slider("Soft Warning Threshold (MFA)", 30, 60, 45)
hard_thresh = st.sidebar.slider("Critical Alert Threshold (Block)", 65, 95, 75)
scorer.soft_threshold = soft_thresh
scorer.hard_threshold = hard_thresh

st.sidebar.markdown("---")
st.sidebar.markdown("**System Health:** `ONLINE`")
st.sidebar.markdown(f"**Compute Device:** `{detector.device.upper()}`")

if st.sidebar.button("🔄 Force Reload Detector Model"):
    st.session_state.detector = VoiceSpoofDetector()
    st.session_state.scorer = DynamicRiskScorer()
    st.rerun()

tab1, tab2 = st.tabs(["📁 File Upload Analysis", "🎙️ Live Mic Test"])

def render_risk_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Dynamic Impersonation Risk Score", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#1f2937"},
            'steps': [
                {'range': [0, soft_thresh], 'color': "#10b981"},
                {'range': [soft_thresh, hard_thresh], 'color': "#f59e0b"},
                {'range': [hard_thresh, 100], 'color': "#ef4444"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# Tab 1: File Upload
with tab1:
    st.subheader("Analyze Recorded Audio Stream")
    uploaded_file = st.file_uploader("Upload incoming call recording (.wav, .mp3)", type=["wav", "mp3"])

    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format="audio/wav")

        with st.spinner("Processing acoustic artifacts and neural voice patterns..."):
            scorer.reset()
            y = extractor.load_audio_bytes(audio_bytes)
            prosody = extractor.extract_prosodic_features(y)
            spoof_prob = detector.predict_spoof_probability(y)
            result = scorer.calculate_risk(spoof_prob, prosody)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.plotly_chart(render_risk_gauge(result["smoothed_risk_score"]), width="stretch", key="gauge_upload")

        with col2:
            st.markdown("### Decision Status")
            if result["status"] == "CRITICAL_ALERT":
                st.error(f"🚨 **{result['status']}**")
                st.markdown(f"**Recommendation:** {result['message']}")
                st.button("⛔ Auto-Block High-Value Transaction", type="primary", key="btn_block_file")
            
            elif result["status"] == "SUSPICIOUS":
                st.warning(f"⚠️ **{result['status']}**")
                st.markdown(f"**Recommendation:** {result['message']}")
                
                if st.button("📲 Trigger Immediate Out-of-Band OTP", key="btn_otp_file"):
                    response = st.session_state.otp_service.generate_and_send_otp(st.session_state.registered_phone)
                    st.session_state.otp_sent = True
                    st.session_state.last_demo_otp = response["demo_otp_display"]
                    st.toast(f"Out-of-Band SMS Sent to {st.session_state.registered_phone}!", icon="📲")

                if st.session_state.otp_sent:
                    st.info(f"🔑 Security OTP dispatched to registered device: `{st.session_state.registered_phone}`")
                    with st.expander("🛠️ LEA / Evaluator Simulation Console"):
                        st.code(f"Simulated SMS Payload received on device:\nOTP: {st.session_state.get('last_demo_otp', '123456')}")

                    with st.form("oob_verification_form"):
                        entered_otp = st.text_input("Enter 6-digit OTP received via SMS:", max_chars=6)
                        submit_btn = st.form_submit_button("Verify Identity & Authorize Call")

                        if submit_btn:
                            verify_res = st.session_state.otp_service.verify_otp(
                                st.session_state.registered_phone, 
                                entered_otp
                            )
                            if verify_res["success"]:
                                st.success(f"✅ {verify_res['message']}")
                                st.balloons()
                                st.session_state.otp_sent = False
                            else:
                                st.error(f"❌ {verify_res['message']}")
            else:
                st.success(f"✅ **{result['status']}**")
                st.markdown(f"**Recommendation:** {result['message']}")

            st.markdown("---")
            st.markdown(f"**Raw Model Spoof Probability:** `{spoof_prob * 100:.2f}%`")
            st.markdown(f"**Pitch Variance:** `{prosody.get('pitch_variance', 0.0):.2f}`")
            st.markdown(f"**Spectral Flatness:** `{prosody.get('spectral_flatness', 0.0):.4f}`")

        st.markdown("### 🔬 Spectral Signature Inspection")
        mel_spec = extractor.compute_mel_spectrogram(y)
        fig_spec, ax = plt.subplots(figsize=(10, 2.8))
        ax.imshow(mel_spec, aspect='auto', origin='lower', cmap='magma')
        ax.set_title("Log-Mel Spectrogram (Acoustic Footprint)")
        ax.set_ylabel("Freq Bins")
        ax.set_xlabel("Time Frames")
        st.pyplot(fig_spec)

# Tab 2: Live Mic Input
with tab2:
    st.subheader("Real-Time Call Simulation via Microphone")
    audio_input = st.audio_input("Record voice sample to simulate ongoing incoming call")

    if audio_input is not None:
        raw_bytes = audio_input.read()
        y_live = extractor.load_audio_bytes(raw_bytes)

        prosody_live = extractor.extract_prosodic_features(y_live)
        prob_live = detector.predict_spoof_probability(y_live)
        live_result = scorer.calculate_risk(prob_live, prosody_live)

        st.plotly_chart(render_risk_gauge(live_result["smoothed_risk_score"]), width="stretch", key="gauge_live")

        if live_result["status"] == "CRITICAL_ALERT":
            st.error(f"🚨 Cloned Voice Signature Detected! Risk: {live_result['smoothed_risk_score']}%")
            st.button("⛔ Terminate Suspicious Call Channel", type="primary", key="btn_block_live")
        elif live_result["status"] == "SUSPICIOUS":
            st.warning(f"⚠️ Ambiguous Audio Signature! Risk: {live_result['smoothed_risk_score']}%")
            st.button("📲 Trigger Interactive Liveness Challenge", key="btn_challenge_live")
        else:
            st.success(f"✅ Natural Biological Voice Confirmed! Risk: {live_result['smoothed_risk_score']}%")
