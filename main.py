import streamlit as st # type: ignore
import os
import json
import tempfile
import uuid
import threading
import wave
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import time

# Core libraries
import whisper # type: ignore
import torch # type: ignore
import numpy as np # type: ignore
import librosa # type: ignore
import soundfile as sf # type: ignore
import ffmpeg # type: ignore
from scipy.signal import resample # type: ignore
import pyaudio # type: ignore

# ML libraries
from transformers import ( # type: ignore
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    pipeline
)

# Database
from sqlalchemy import ( # type: ignore
    create_engine, Column, Integer, String, 
    DateTime, Text, Boolean, Float
)
# from sqlalchemy.ext.declarative import declarative_base # type: ignore
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session # type: ignore

# Security
from passlib.context import CryptContext # type: ignore
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
SUPPORTED_LANGUAGES = {
    "auto": "Auto-detect",
    "de": "German", 
    "gsw": "Swiss German",
    "fr": "French",
    "it": "Italian",
    "en": "English"
}
AUDIO_SAMPLE_RATE = 16000
CHUNK_SIZE = 1024

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./offline_transcriber.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Security setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Meeting(Base):
    __tablename__ = "meetings"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True, nullable=False)
    user_id = Column(Integer, index=True, nullable=False)
    transcript = Column(Text)  # JSON string of segments
    summary = Column(Text)
    speakers = Column(Text)  # JSON string of speaker info
    audio_file_path = Column(String)
    language = Column(String, default="auto")
    detected_language = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    duration = Column(Float)  # in seconds
    status = Column(String, default="processing")  # processing, completed, failed

# Create tables
Base.metadata.create_all(bind=engine)

# AI Models Service
class OfflineAIService:
    def __init__(self):
        self.whisper_model = None
        self.summarization_pipeline = None
        self.models_loaded = False
        
    @st.cache_resource
    def load_models(_self):
        """Load all AI models for offline operation"""
        try:
            with st.spinner("Loading Whisper model..."):
                _self.whisper_model = whisper.load_model("base")
            
            with st.spinner("Loading summarization model..."):
                try:
                    _self.summarization_pipeline = pipeline(
                        "summarization",
                        model="sshleifer/distilbert-base-cnn-12-6",
                        device=0 if torch.cuda.is_available() else -1
                    )
                except Exception as e:
                    logger.warning(f"Could not load summarization model: {e}")
                    _self.summarization_pipeline = None
            
            _self.models_loaded = True
            st.success("All AI models loaded successfully!")
            return _self
            
        except Exception as e:
            st.error(f"Error loading models: {e}")
            raise e
    
    def transcribe_audio(self, audio_path: str, language: str = "auto") -> Dict[str, Any]:
        """Transcribe audio using Whisper"""
        if not self.models_loaded:
            raise RuntimeError("Models not loaded")
        
        try:
            # Transcribe with Whisper
            if language == "auto":
                result = self.whisper_model.transcribe(
                    audio_path,
                    fp16=False,
                    verbose=False
                )
            else:
                result = self.whisper_model.transcribe(
                    audio_path,
                    language=language,
                    fp16=False,
                    verbose=False
                )
            
            return {
                "text": result["text"],
                "segments": result["segments"],
                "language": result["language"]
            }
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise e
    
    def perform_speaker_diarization(self, audio_path: str) -> Dict[str, List[Dict]]:
        """Mock speaker diarization for demo"""
        try:
            # Get audio duration
            audio, sr = librosa.load(audio_path, sr=None)
            duration = len(audio) / sr
            
            # Create mock speaker segments
            mid_point = duration / 2
            speakers = {
                "SPEAKER_00": [{"start": 0, "end": mid_point}],
                "SPEAKER_01": [{"start": mid_point, "end": duration}]
            }
            
            return speakers
        except Exception as e:
            logger.error(f"Diarization error: {e}")
            return {
                "SPEAKER_00": [{"start": 0, "end": 30}],
                "SPEAKER_01": [{"start": 30, "end": 60}]
            }
    
    def combine_transcript_with_speakers(
        self, 
        transcript: Dict[str, Any], 
        speakers: Dict[str, List[Dict]]
    ) -> List[Dict]:
        """Combine transcription with speaker diarization"""
        combined_segments = []
        
        for segment in transcript["segments"]:
            segment_start = segment["start"]
            segment_end = segment["end"]
            segment_text = segment["text"].strip()
            
            # Find the speaker with maximum overlap
            best_speaker = "UNKNOWN"
            max_overlap = 0
            
            for speaker, intervals in speakers.items():
                for interval in intervals:
                    # Calculate overlap
                    overlap_start = max(segment_start, interval["start"])
                    overlap_end = min(segment_end, interval["end"])
                    overlap = max(0, overlap_end - overlap_start)
                    
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_speaker = speaker
            
            combined_segments.append({
                "text": segment_text,
                "start": segment_start,
                "end": segment_end,
                "speaker": best_speaker,
                "confidence": segment.get("avg_confidence", 0.8)
            })
        
        return combined_segments
    
    def summarize_transcript(self, transcript_text: str) -> str:
        """Generate summary using local LLM"""
        if not transcript_text.strip():
            return "No content to summarize."
        
        try:
            if self.summarization_pipeline:
                summary = self.summarization_pipeline(
                    transcript_text,
                    max_length=150,
                    min_length=50,
                    do_sample=False
                )
                return summary[0]["summary_text"]
            else:
                # Fallback: simple extractive summary
                sentences = transcript_text.split('. ')
                if len(sentences) <= 3:
                    return transcript_text
                return '. '.join(sentences[:3]) + '.'
                
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            # Fallback: simple extractive summary
            sentences = transcript_text.split('. ')
            if len(sentences) <= 3:
                return transcript_text
            return '. '.join(sentences[:3]) + '.'

# Audio Processing Utilities
class AudioProcessor:
    @staticmethod
    def convert_to_wav(input_path: str, output_path: str) -> str:
        """Convert audio/video file to WAV format"""
        try:
            (
                ffmpeg
                .input(input_path)
                .output(
                    output_path,
                    acodec='pcm_s16le',
                    ac=1,
                    ar=AUDIO_SAMPLE_RATE
                )
                .overwrite_output()
                .run(quiet=True)
            )
            return output_path
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            # Fallback using librosa
            audio, sr = librosa.load(input_path, sr=AUDIO_SAMPLE_RATE)
            sf.write(output_path, audio, AUDIO_SAMPLE_RATE)
            return output_path
    
    @staticmethod
    def get_audio_duration(file_path: str) -> float:
        """Get audio duration in seconds"""
        try:
            audio, sr = librosa.load(file_path, sr=None)
            return len(audio) / sr
        except Exception as e:
            logger.error(f"Could not get audio duration: {e}")
            return 0.0

# Real-time Audio Recording
class RealTimeRecorder:
    def __init__(self):
        self.is_recording = False
        self.audio_data = []
        self.sample_rate = AUDIO_SAMPLE_RATE
        self.chunk = CHUNK_SIZE
        
    def start_recording(self):
        """Start recording audio from microphone"""
        self.is_recording = True
        self.audio_data = []
        
        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            
            while self.is_recording:
                try:
                    data = stream.read(self.chunk, exception_on_overflow=False)
                    self.audio_data.append(data)
                except Exception as e:
                    logger.error(f"Recording error: {e}")
                    break
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
        except Exception as e:
            st.error(f"Could not start recording: {e}")
            self.is_recording = False
    
    def stop_recording(self) -> Optional[str]:
        """Stop recording and save audio file"""
        self.is_recording = False
        
        if not self.audio_data:
            return None
        
        # Save audio to temporary file
        temp_file = tempfile.NamedTemporaryFile(
            suffix=".wav", 
            delete=False
        )
        
        try:
            wf = wave.open(temp_file.name, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.audio_data))
            wf.close()
            
            return temp_file.name
        except Exception as e:
            st.error(f"Could not save recording: {e}")
            return None

# Database utilities
def get_db():
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

# Initialize AI service
@st.cache_resource
def get_ai_service():
    service = OfflineAIService()
    return service.load_models()

# Streamlit App
def main():
    st.set_page_config(
        page_title="Offline Meeting Transcriber",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎤 Offline Meeting Transcriber")
    st.markdown("AI-powered meeting transcription with speaker diarization - completely offline")
    
    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    if "ai_service" not in st.session_state:
        st.session_state.ai_service = None
    
    # Load AI models if not loaded
    if st.session_state.ai_service is None:
        st.session_state.ai_service = get_ai_service()
    
    # Authentication
    if not st.session_state.authenticated:
        show_auth_page()
    else:
        show_main_app()

def show_auth_page():
    st.markdown("### 🔐 Authentication")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                db = get_db()
                user = db.query(User).filter(User.username == username).first()
                
                if user and verify_password(password, user.hashed_password):
                    st.session_state.authenticated = True
                    st.session_state.current_user = user
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    
    with tab2:
        st.subheader("Register")
        with st.form("register_form"):
            reg_username = st.text_input("Username", key="reg_username")
            reg_email = st.text_input("Email", key="reg_email")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            reg_password_confirm = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Register")
            
            if submitted:
                if reg_password != reg_password_confirm:
                    st.error("Passwords do not match")
                elif len(reg_password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    db = get_db()
                    
                    # Check if user exists
                    existing_user = db.query(User).filter(
                        (User.username == reg_username) | (User.email == reg_email)
                    ).first()
                    
                    if existing_user:
                        st.error("Username or email already exists")
                    else:
                        # Create new user
                        hashed_password = get_password_hash(reg_password)
                        new_user = User(
                            username=reg_username,
                            email=reg_email,
                            hashed_password=hashed_password
                        )
                        db.add(new_user)
                        db.commit()
                        db.refresh(new_user)
                        
                        st.success("Registration successful! Please login.")

def show_main_app():
    # Sidebar
    with st.sidebar:
        st.markdown(f"### Welcome, {st.session_state.current_user.username}")
        
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.current_user = None
            st.rerun()
        
        st.markdown("---")
        
        # Navigation
        page = st.selectbox(
            "Navigate",
            ["Upload & Transcribe", "Real-time Recording", "My Meetings", "Settings"]
        )
    
    # Main content
    if page == "Upload & Transcribe":
        show_upload_page()
    elif page == "Real-time Recording":
        show_realtime_page()
    elif page == "My Meetings":
        show_meetings_page()
    elif page == "Settings":
        show_settings_page()

def show_upload_page():
    st.header("📁 Upload & Transcribe")
    
    with st.form("upload_form"):
        title = st.text_input("Meeting Title", placeholder="Enter meeting title...")
        language = st.selectbox("Language", list(SUPPORTED_LANGUAGES.keys()), 
                               format_func=lambda x: SUPPORTED_LANGUAGES[x])
        
        uploaded_file = st.file_uploader(
            "Choose audio/video file",
            type=['wav', 'mp3', 'mp4', 'avi', 'mov', 'flv', 'm4a', 'aac']
        )
        
        submitted = st.form_submit_button("Upload & Process")
        
        if submitted and uploaded_file and title:
            process_uploaded_file(uploaded_file, title, language)
        elif submitted:
            st.error("Please provide both title and file")

def process_uploaded_file(uploaded_file, title, language):
    try:
        # Save uploaded file
        file_id = str(uuid.uuid4())
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        # Create uploads directory if it doesn't exist
        Path("uploads").mkdir(exist_ok=True)
        
        upload_path = f"uploads/{file_id}_original{file_extension}"
        with open(upload_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Convert to WAV
        wav_path = f"uploads/{file_id}.wav"
        with st.spinner("Converting audio..."):
            AudioProcessor.convert_to_wav(upload_path, wav_path)
        
        # Get duration
        duration = AudioProcessor.get_audio_duration(wav_path)
        
        # Create meeting record
        db = get_db()
        meeting = Meeting(
            title=title,
            user_id=st.session_state.current_user.id,
            audio_file_path=wav_path,
            language=language,
            duration=duration,
            status="processing"
        )
        db.add(meeting)
        db.commit()
        db.refresh(meeting)
        
        # Process the meeting
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Transcribe
            status_text.text("🎯 Transcribing audio...")
            progress_bar.progress(25)
            transcript_result = st.session_state.ai_service.transcribe_audio(wav_path, language)
            
            # Speaker diarization
            status_text.text("👥 Identifying speakers...")
            progress_bar.progress(50)
            speakers = st.session_state.ai_service.perform_speaker_diarization(wav_path)
            
            # Combine transcript with speakers
            status_text.text("🔄 Combining transcription with speakers...")
            progress_bar.progress(75)
            combined_transcript = st.session_state.ai_service.combine_transcript_with_speakers(
                transcript_result, speakers
            )
            
            # Generate summary
            status_text.text("📝 Generating summary...")
            progress_bar.progress(90)
            full_text = " ".join([seg["text"] for seg in combined_transcript])
            summary = st.session_state.ai_service.summarize_transcript(full_text)
            
            # Update database
            meeting.transcript = json.dumps(combined_transcript)
            meeting.summary = summary
            meeting.speakers = json.dumps(list(speakers.keys()))
            meeting.detected_language = transcript_result["language"]
            meeting.status = "completed"
            db.commit()
            
            progress_bar.progress(100)
            status_text.text("✅ Processing completed!")
            
            # Clean up original file
            if os.path.exists(upload_path):
                os.unlink(upload_path)
            
            st.success("Meeting processed successfully!")
            
            # Display results
            show_meeting_results(meeting)
            
        except Exception as e:
            meeting.status = "failed"
            db.commit()
            st.error(f"Processing failed: {str(e)}")
            
    except Exception as e:
        st.error(f"Upload failed: {str(e)}")

def show_realtime_page():
    st.header("🎙️ Real-time Recording")
    
    if "recorder" not in st.session_state:
        st.session_state.recorder = RealTimeRecorder()
    if "recording_thread" not in st.session_state:
        st.session_state.recording_thread = None
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔴 Start Recording", disabled=st.session_state.recorder.is_recording):
            st.session_state.recording_thread = threading.Thread(
                target=st.session_state.recorder.start_recording
            )
            st.session_state.recording_thread.start()
            st.success("Recording started!")
    
    with col2:
        if st.button("⏹️ Stop Recording", disabled=not st.session_state.recorder.is_recording):
            audio_file = st.session_state.recorder.stop_recording()
            
            if audio_file:
                st.success("Recording stopped!")
                
                # Process the recording
                with st.spinner("Transcribing..."):
                    try:
                        result = st.session_state.ai_service.transcribe_audio(audio_file)
                        
                        st.subheader("Transcription Results:")
                        st.write(f"**Detected Language:** {result['language']}")
                        st.write(f"**Text:** {result['text']}")
                        
                        # Optionally save to database
                        if st.button("Save this recording"):
                            title = st.text_input("Enter title for this recording:")
                            if title:
                                # Save to database similar to upload process
                                pass
                        
                        # Clean up temp file
                        os.unlink(audio_file)
                        
                    except Exception as e:
                        st.error(f"Transcription failed: {str(e)}")
            else:
                st.warning("No audio recorded")
    
    # Show recording status
    if st.session_state.recorder.is_recording:
        st.info("🔴 Recording in progress...")

def show_meetings_page():
    st.header("📋 My Meetings")
    
    db = get_db()
    meetings = (
        db.query(Meeting)
        .filter(Meeting.user_id == st.session_state.current_user.id)
        .order_by(Meeting.created_at.desc())
        .all()
    )
    
    if not meetings:
        st.info("No meetings found. Upload your first audio file!")
        return
    
    for meeting in meetings:
        with st.expander(
            f"📄 {meeting.title} - {meeting.status.title()} "
            f"({meeting.created_at.strftime('%Y-%m-%d %H:%M')})"
        ):
            show_meeting_details(meeting)

def show_meeting_details(meeting):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Duration", f"{meeting.duration:.1f}s" if meeting.duration else "N/A")
    with col2:
        st.metric("Language", meeting.detected_language or meeting.language)
    with col3:
        if st.button("🗑️ Delete", key=f"delete_{meeting.id}"):
            delete_meeting(meeting)
    
    if meeting.status == "completed":
        show_meeting_results(meeting)

def show_meeting_results(meeting):
    if meeting.summary:
        st.subheader("📝 Summary")
        st.write(meeting.summary)
    
    if meeting.transcript:
        st.subheader("📜 Full Transcript")
        transcript = json.loads(meeting.transcript)
        
        for segment in transcript:
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "")
            start_time = segment.get("start", 0)
            
            # Format time
            minutes = int(start_time // 60)
            seconds = int(start_time % 60)
            
            st.markdown(
                f"**{speaker}** [{minutes:02d}:{seconds:02d}]: {text}"
            )

def delete_meeting(meeting):
    try:
        db = get_db()
        
        # Delete audio file
        if meeting.audio_file_path and os.path.exists(meeting.audio_file_path):
            os.unlink(meeting.audio_file_path)
        
        # Delete from database
        db.delete(meeting)
        db.commit()
        
        st.success("Meeting deleted successfully!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Failed to delete meeting: {str(e)}")

def show_settings_page():
    st.header("⚙️ Settings")
    
    st.subheader("Supported Languages")
    for code, name in SUPPORTED_LANGUAGES.items():
        st.write(f"**{code}**: {name}")
    
    st.subheader("System Information")
    st.write(f"**CUDA Available**: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        st.write(f"**GPU**: {torch.cuda.get_device_name()}")
    
    st.write(f"**Models Loaded**: {st.session_state.ai_service.models_loaded}")
    
    if st.button("Clear Cache & Reload Models"):
        st.cache_resource.clear()
        st.session_state.ai_service = None
        st.rerun()

if __name__ == "__main__":
    main()