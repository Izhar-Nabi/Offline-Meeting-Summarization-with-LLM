import streamlit as st
import requests
import time
import os
import sqlite3
import hashlib
import jwt
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import pyaudio
import wave
import threading
import tempfile
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")

# Load .env for GROQ_API_KEY
load_dotenv()

# Configuration
DATABASE_PATH = "users.db"
SECRET_KEY = "put database API key here..."
LOG_FILE = "transcription_app.log"

# Language codes supported by AssemblyAI
SUPPORTED_LANGUAGES = {
    "Swiss German": "de-CH",  # Primary language you mentioned
    "German": "de",
    "French": "fr", 
    "Italian": "it",
    "English": "en",
    "Spanish": "es", 
    "Portuguese": "pt",
    "Dutch": "nl",
    "Hindi": "hi",
    "Japanese": "ja",
    "Chinese": "zh",
    "Korean": "ko",
    "Russian": "ru",
    "Arabic": "ar",
    "Turkish": "tr",
    "Vietnamese": "vi",
    "Ukrainian": "uk",
    "Hebrew": "he",
    "Polish": "pl",
    "Swedish": "sv",
    "Norwegian": "no",
    "Danish": "da",
    "Finnish": "fi",
    "Greek": "el",
    "Czech": "cs",
    "Hungarian": "hu",
    "Romanian": "ro",
    "Slovak": "sk",
    "Bulgarian": "bg",
    "Croatian": "hr",
    "Serbian": "sr",
    "Slovenian": "sl",
    "Estonian": "et",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Maltese": "mt",
    "Thai": "th",
    "Malay": "ms",
    "Indonesian": "id",
    "Tagalog": "tl",
    "Swahili": "sw",
    "Yoruba": "yo",
    "Zulu": "zu",
    "Afrikaans": "af"
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# AssemblyAI setup
BASE_URL = "https://api.assemblyai.com"
API_KEY = "# Replace with your API key"

HEADERS = {
    "authorization": API_KEY
}

class DatabaseManager:
    def __init__(self, db_path=DATABASE_PATH):
        self.db_path = db_path
        self.init_database()
        logger.info("Database manager initialized")
    
    def init_database(self):
        """Initialize the user database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transcription_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                filename TEXT,
                language TEXT,
                transcript_text TEXT,
                summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database tables created/verified")
    
    def create_user(self, username, password, email=None):
        """Create a new user"""
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)", 
                         (username, password_hash, email))
            user_id = cursor.lastrowid
            conn.commit()
            conn.close()
            logger.info(f"User created: {username}")
            return True
        except sqlite3.IntegrityError:
            logger.warning(f"Failed to create user - username already exists: {username}")
            return False
    
    def authenticate_user(self, username, password):
        """Authenticate user and update last login"""
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ? AND password_hash = ?", 
                      (username, password_hash))
        result = cursor.fetchone()
        
        if result:
            # Update last login
            cursor.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE username = ?", 
                          (username,))
            conn.commit()
            logger.info(f"User authenticated: {username}")
        else:
            logger.warning(f"Failed authentication attempt: {username}")
        
        conn.close()
        return result is not None
    
    def get_user_id(self, username):
        """Get user ID by username"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    
    def save_transcription(self, user_id, filename, language, transcript_text, summary=None):
        """Save transcription to history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO transcription_history (user_id, filename, language, transcript_text, summary)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, filename, language, transcript_text, summary))
        conn.commit()
        conn.close()
        logger.info(f"Transcription saved for user {user_id}: {filename}")
    
    def get_user_history(self, user_id, limit=10):
        """Get user's transcription history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT filename, language, created_at, transcript_text, summary
            FROM transcription_history 
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (user_id, limit))
        result = cursor.fetchall()
        conn.close()
        return result

class LocalSummarizer:
    def __init__(self):
        self.summarizer = None
        self.load_local_model()
    
    @st.cache_resource
    def load_local_model(_self):
        """Load local summarization model"""
        try:
            # Using a lightweight model for summarization
            model_name = "facebook/bart-large-cnn"
            _self.summarizer = pipeline("summarization", model=model_name, device=-1)
            logger.info("Local summarization model loaded")
            return True
        except Exception as e:
            logger.error(f"Failed to load local summarization model: {e}")
            return False
    
    def summarize_text(self, text, max_length=200):
        """Summarize text using local model"""
        if not self.summarizer:
            logger.warning("Local summarizer not available, falling back to Groq")
            return self.summarize_with_groq(text)
        
        try:
            # Split text into chunks if too long
            max_chunk_length = 1000
            if len(text) > max_chunk_length:
                chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
                summaries = []
                for chunk in chunks:
                    summary = self.summarizer(chunk, max_length=max_length//len(chunks), min_length=30, do_sample=False)
                    summaries.append(summary[0]['summary_text'])
                return " ".join(summaries)
            else:
                summary = self.summarizer(text, max_length=max_length, min_length=30, do_sample=False)
                return summary[0]['summary_text']
        except Exception as e:
            logger.error(f"Local summarization failed: {e}")
            return self.summarize_with_groq(text)
    
    def summarize_with_groq(self, text):
        """Fallback to Groq summarization"""
        try:
            groq_api_key = os.environ.get("GROQ_API_KEY")
            if not groq_api_key:
                return "❌ Both local and Groq summarization unavailable"
            
            llm = ChatGroq(
                temperature=0.7,
                model_name="llama3-8b-8192",
                api_key=groq_api_key
            )
            
            prompt = ChatPromptTemplate.from_template(
                "Summarize the following transcript in a concise manner:\n\n{transcript}"
            )
            
            chain = prompt | llm | StrOutputParser()
            result = chain.invoke({"transcript": text})
            logger.info("Groq summarization completed")
            return result
        except Exception as e:
            logger.error(f"Groq summarization failed: {e}")
            return f"❌ Summarization failed: {str(e)}"

class AssemblyAITranscriber:
    def __init__(self):
        self.is_recording = False
        logger.info("AssemblyAI transcriber initialized")
    
    def upload_audio(self, file):
        """Upload audio file to AssemblyAI"""
        try:
            response = requests.post(
                f"{BASE_URL}/v2/upload",
                headers=HEADERS,
                data=file
            )
            response.raise_for_status()
            upload_url = response.json()["upload_url"]
            logger.info("Audio file uploaded to AssemblyAI")
            return upload_url
        except Exception as e:
            logger.error(f"Failed to upload audio: {e}")
            raise
    
    def request_transcription(self, audio_url, language_code=None, speaker_labels=False):
        """Request transcription with language and speaker options"""
        data = {
            "audio_url": audio_url,
            "speaker_labels": speaker_labels,
            "speakers_expected": 2 if speaker_labels else None
        }
        
        # Add language code if specified
        if language_code and language_code != "auto":
            data["language_code"] = language_code
        
        try:
            response = requests.post(f"{BASE_URL}/v2/transcript", headers=HEADERS, json=data)
            response.raise_for_status()
            transcript_id = response.json()["id"]
            logger.info(f"Transcription requested with ID: {transcript_id}")
            return transcript_id
        except Exception as e:
            logger.error(f"Failed to request transcription: {e}")
            raise
    
    def poll_transcription(self, transcript_id):
        """Poll for transcription completion"""
        polling_endpoint = f"{BASE_URL}/v2/transcript/{transcript_id}"
        logger.info(f"Polling transcription: {transcript_id}")
        
        while True:
            try:
                response = requests.get(polling_endpoint, headers=HEADERS)
                result = response.json()
                
                if result["status"] == "completed":
                    logger.info("Transcription completed successfully")
                    return result
                elif result["status"] == "error":
                    logger.error(f"Transcription failed: {result.get('error', 'Unknown error')}")
                    raise RuntimeError(f"Transcription failed: {result.get('error', 'Unknown error')}")
                else:
                    logger.info(f"Transcription status: {result['status']}")
                    time.sleep(3)
            except Exception as e:
                logger.error(f"Error polling transcription: {e}")
                raise
    
    def format_transcript_with_speakers(self, result):
        """Format transcript with speaker labels"""
        if not result.get("utterances"):
            return result.get("text", "")
        
        formatted_text = ""
        for utterance in result["utterances"]:
            speaker = utterance.get("speaker", "Unknown")
            text = utterance.get("text", "")
            start_time = utterance.get("start", 0) / 1000  # Convert to seconds
            formatted_text += f"[Speaker {speaker}] ({start_time:.1f}s): {text}\n\n"
        
        return formatted_text
    
    def start_real_time_recording(self):
        """Start real-time audio recording"""
        self.is_recording = True
        logger.info("Real-time recording started")
        
        # Audio recording parameters
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        
        audio = pyaudio.PyAudio()
        
        try:
            stream = audio.open(format=FORMAT,
                              channels=CHANNELS,
                              rate=RATE,
                              input=True,
                              frames_per_buffer=CHUNK)
            
            frames = []
            while self.is_recording:
                data = stream.read(CHUNK)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            
            # Save recorded audio to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                wf = wave.open(tmp_file.name, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                
                logger.info("Real-time recording saved to temporary file")
                return tmp_file.name
                
        except Exception as e:
            logger.error(f"Real-time recording failed: {e}")
            raise
        finally:
            audio.terminate()
    
    def stop_recording(self):
        """Stop real-time recording"""
        self.is_recording = False
        logger.info("Real-time recording stopped")

def create_jwt_token(username):
    """Create JWT token for user session"""
    payload = {
        'username': username,
        'exp': datetime.utcnow() + timedelta(hours=24)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')

def verify_jwt_token(token):
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload['username']
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token expired")
        return None
    except jwt.InvalidTokenError:
        logger.warning("Invalid JWT token")
        return None

def main():
    st.set_page_config(
        page_title="Enhanced Audio Transcription", 
        page_icon="🎧",
        layout="wide"
    )
    
    # Initialize components
    db_manager = DatabaseManager()
    transcriber = AssemblyAITranscriber()
    summarizer = LocalSummarizer()
    
    # Session state management
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'real_time_text' not in st.session_state:
        st.session_state.real_time_text = ""
    if 'recording_file' not in st.session_state:
        st.session_state.recording_file = None
    
    # Authentication Section
    if not st.session_state.authenticated:
        st.title("🔐 Authentication")
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.subheader("Login")
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Login", type="primary"):
                    if username and password:
                        if db_manager.authenticate_user(username, password):
                            st.session_state.authenticated = True
                            st.session_state.username = username
                            st.success("✅ Login successful!")
                            logger.info(f"User logged in: {username}")
                            st.rerun()
                        else:
                            st.error("❌ Invalid credentials")
                    else:
                        st.error("❌ Please enter username and password")
        
        with tab2:
            st.subheader("Register New Account")
            new_username = st.text_input("Username", key="register_username")
            new_email = st.text_input("Email (optional)", key="register_email")
            new_password = st.text_input("Password", type="password", key="register_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            
            if st.button("Register", type="primary"):
                if not new_username or not new_password:
                    st.error("❌ Username and password are required")
                elif new_password != confirm_password:
                    st.error("❌ Passwords don't match")
                elif len(new_password) < 6:
                    st.error("❌ Password must be at least 6 characters")
                elif db_manager.create_user(new_username, new_password, new_email):
                    st.success("✅ Registration successful! Please login.")
                else:
                    st.error("❌ Username already exists")
    
    else:
        # Main Application
        st.title("🎧 Enhanced Audio Transcription & Summarization")
        
        # Sidebar
        with st.sidebar:
            st.write(f"👤 **Welcome, {st.session_state.username}!**")
            
            if st.button("🚪 Logout", type="secondary"):
                st.session_state.authenticated = False
                st.session_state.username = None
                logger.info(f"User logged out: {st.session_state.username}")
                st.rerun()
            
            st.divider()
            
            # User Statistics
            user_id = db_manager.get_user_id(st.session_state.username)
            if user_id:
                history = db_manager.get_user_history(user_id, 5)
                st.subheader("📊 Recent Activity")
                st.write(f"**Total Transcriptions:** {len(history)}")
                
                if history:
                    st.write("**Recent Files:**")
                    for item in history[:3]:
                        st.write(f"• {item[0]} ({item[1]})")
        
        # Main Content Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📁 File Upload", "🎤 Real-time", "📚 History", "⚙️ Settings"])
        
        with tab1:
            st.header("Upload Audio/Video File")
            
            col1, col2 = st.columns(2)
            
            with col1:
                language = st.selectbox(
                    "Select Language", 
                    ["Auto-detect"] + list(SUPPORTED_LANGUAGES.keys()),
                    help="Choose the language of your audio file"
                )
            
            with col2:
                enable_speakers = st.checkbox(
                    "Enable Speaker Diarization", 
                    value=False,
                    help="Identify different speakers in the audio"
                )
            
            uploaded_file = st.file_uploader(
                "Upload your audio/video file:", 
                type=["mp3", "wav", "m4a", "mp4", "avi", "mov", "flac", "ogg"],
                help="Supported formats: MP3, WAV, M4A, MP4, AVI, MOV, FLAC, OGG"
            )
            
            if uploaded_file is not None:
                st.success(f"✅ File uploaded: {uploaded_file.name}")
                
                if st.button("🚀 Start Transcription", type="primary"):
                    try:
                        with st.spinner("📤 Uploading audio..."):
                            audio_url = transcriber.upload_audio(uploaded_file)
                        
                        with st.spinner("🧾 Requesting transcription..."):
                            language_code = SUPPORTED_LANGUAGES.get(language, None) if language != "Auto-detect" else None
                            transcript_id = transcriber.request_transcription(
                                audio_url, 
                                language_code=language_code,
                                speaker_labels=enable_speakers
                            )
                        
                        with st.spinner("📝 Transcribing audio... please wait."):
                            result = transcriber.poll_transcription(transcript_id)
                        
                        # Format transcript
                        if enable_speakers:
                            transcript_text = transcriber.format_transcript_with_speakers(result)
                        else:
                            transcript_text = result["text"]
                        
                        st.subheader("📝 Transcript")
                        st.text_area("Transcript", transcript_text, height=300)
                        
                        # Auto-generate summary
                        st.subheader("📄 Summary")
                        with st.spinner("🤖 Generating summary..."):
                            summary = summarizer.summarize_text(transcript_text)
                        
                        st.text_area("Summary", summary, height=200)
                        
                        # Save to database
                        user_id = db_manager.get_user_id(st.session_state.username)
                        if user_id:
                            db_manager.save_transcription(
                                user_id, 
                                uploaded_file.name, 
                                language, 
                                transcript_text, 
                                summary
                            )
                        
                        st.success("✅ Transcription completed and saved!")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        logger.error(f"Transcription error: {e}")
        
        with tab2:
            st.header("🎤 Real-time Transcription")
            
            language_rt = st.selectbox(
                "Select Language for Real-time", 
                ["Auto-detect"] + list(SUPPORTED_LANGUAGES.keys()),
                key="rt_lang"
            )
            
            enable_speakers_rt = st.checkbox(
                "Enable Speaker Diarization for Real-time", 
                value=False,
                key="rt_speakers"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🎤 Start Recording", type="primary"):
                    if not transcriber.is_recording:
                        st.info("🎤 Recording started... Click 'Stop Recording' to end.")
                        
                        def record_and_transcribe():
                            try:
                                # Record audio
                                audio_file_path = transcriber.start_real_time_recording()
                                
                                # Upload and transcribe
                                with open(audio_file_path, 'rb') as f:
                                    audio_url = transcriber.upload_audio(f)
                                
                                language_code = SUPPORTED_LANGUAGES.get(language_rt, None) if language_rt != "Auto-detect" else None
                                transcript_id = transcriber.request_transcription(
                                    audio_url,
                                    language_code=language_code,
                                    speaker_labels=enable_speakers_rt
                                )
                                
                                result = transcriber.poll_transcription(transcript_id)
                                
                                if enable_speakers_rt:
                                    transcript_text = transcriber.format_transcript_with_speakers(result)
                                else:
                                    transcript_text = result["text"]
                                
                                st.session_state.real_time_text = transcript_text
                                
                                # Clean up
                                os.unlink(audio_file_path)
                                
                            except Exception as e:
                                st.error(f"Recording failed: {str(e)}")
                                logger.error(f"Real-time recording error: {e}")
                        
                        # Start recording in a separate thread
                        thread = threading.Thread(target=record_and_transcribe)
                        thread.daemon = True
                        thread.start()
            
            with col2:
                if st.button("⏹️ Stop Recording", type="secondary"):
                    transcriber.stop_recording()
                    st.success("🎤 Recording stopped!")
            
            # Display real-time results
            if st.session_state.real_time_text:
                st.subheader("📝 Real-time Transcript")
                st.text_area("Real-time Transcript", st.session_state.real_time_text, height=200)
                
                if st.button("Generate Summary", key="rt_summary"):
                    with st.spinner("🤖 Generating summary..."):
                        summary = summarizer.summarize_text(st.session_state.real_time_text)
                        st.text_area("Real-time Summary", summary, height=150)
                        
                        # Save real-time transcription
                        user_id = db_manager.get_user_id(st.session_state.username)
                        if user_id:
                            db_manager.save_transcription(
                                user_id,
                                f"Real-time_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                language_rt,
                                st.session_state.real_time_text,
                                summary
                            )
        
        with tab3:
            st.header("📚 Transcription History")
            
            user_id = db_manager.get_user_id(st.session_state.username)
            if user_id:
                history = db_manager.get_user_history(user_id, 20)
                
                if history:
                    for i, (filename, language, created_at, transcript, summary) in enumerate(history):
                        with st.expander(f"📄 {filename} - {language} ({created_at})"):
                            st.subheader("Transcript")
                            st.text_area("", transcript, height=150, key=f"hist_trans_{i}")
                            
                            if summary:
                                st.subheader("Summary")
                                st.text_area("", summary, height=100, key=f"hist_sum_{i}")
                else:
                    st.info("No transcription history found.")
        
        with tab4:
            st.header("⚙️ Settings & Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🌐 Supported Languages")
                st.info(f"**{len(SUPPORTED_LANGUAGES)} languages supported** including:")
                
                # Highlight the primary languages mentioned
                st.markdown("**Primary Languages:**")
                st.write("• Swiss German (de-CH)")
                st.write("• German (de)")
                st.write("• French (fr)")
                st.write("• Italian (it)")
                
                st.markdown("**Other Supported Languages:**")
                # Display other languages in columns
                other_langs = [lang for lang in SUPPORTED_LANGUAGES.keys() 
                             if lang not in ["Swiss German", "German", "French", "Italian"]]
                for i in range(0, len(other_langs), 3):
                    lang_row = other_langs[i:i+3]
                    st.write(" • ".join(lang_row))
            
            with col2:
                st.subheader("📊 System Status")
                st.info("🟢 AssemblyAI: Connected")
                
                # Check Groq status
                groq_status = "🟢 Available" if os.environ.get("GROQ_API_KEY") else "🔴 Not configured"
                st.info(f"🤖 Groq LLM: {groq_status}")
                
                # Check local summarizer
                local_sum_status = "🟢 Available" if summarizer.summarizer else "🔴 Not available"
                st.info(f"📝 Local Summarizer: {local_sum_status}")
            
            st.subheader("🔧 Features")
            st.markdown("""
            - **Multi-language Support**: 40+ languages with special focus on:
              - **Swiss German** (de-CH)
              - **German** (de)
              - **French** (fr)
              - **Italian** (it)
            - **Speaker Diarization**: Identify different speakers ("who said what")
            - **Real-time Transcription**: Live audio recording and transcription
            - **Local Summarization**: Fallback to Groq if needed
            - **User Authentication**: Secure login with JWT tokens
            - **Transcription History**: Save and review past transcriptions
            - **Comprehensive Logging**: All actions are logged for debugging
            - **Offline Hosting Ready**: Can be configured for local-only operation
            """)
            
            st.subheader("📋 Activity Logs")
            if st.button("View Recent Logs"):
                try:
                    with open(LOG_FILE, 'r') as f:
                        logs = f.readlines()[-20:]  # Show last 20 lines
                    st.text_area("Recent Logs", "".join(logs), height=200)
                except Exception as e:
                    st.error(f"Could not read logs: {e}")

if __name__ == "__main__":
    main()