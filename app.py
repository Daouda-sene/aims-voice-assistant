import streamlit as st
from faster_whisper import WhisperModel
from src.pipeline import ask_question
from gtts import gTTS
from audiorecorder import audiorecorder
import os
from gtts import gTTS
# =====================================
# CONFIG
# =====================================

st.set_page_config(
    page_title="AIMS Senegal Voice Agent",
    page_icon="🎤",
    layout="wide"
)

# =====================================
# STYLE
# =====================================

st.markdown("""
<style>

.title {
    color:#7A1E0E;
    font-size:40px;
    font-weight:bold;
}

.subtitle {
    color:#003366;
    font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# SIDEBAR
# =====================================

st.sidebar.image(
    "assets/aims_logo.png",
    width="stretch"
)

st.sidebar.title("AIMS Voice Agent")

st.sidebar.markdown("""
### Technologies

- 🎤 Faster Whisper
- 📚 ChromaDB
- 🔎 MiniLM
- 🤖 Phi-3
- 🔊 gTTS

### Project

Local Voice Agent with RAG

GAAI-AIMS Final Project
""")

# =====================================
# HEADER
# =====================================

col1, col2 = st.columns([1,5])

with col1:
    st.image(
        "assets/aims_logo.png",
        width=120
    )

with col2:

    st.markdown(
        '<p class="title">AIMS Senegal Voice Assistant</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">Whisper + RAG + Phi-3 + TTS</p>',
        unsafe_allow_html=True
    )

st.divider()

# =====================================
# KPI
# =====================================

c1, c2, c3, c4 = st.columns(4)

c1.metric("PDFs", "9")
c2.metric("Pages", "320")
c3.metric("Chunks", "496")
c4.metric("LLM", "Phi-3")

st.divider()

# =====================================
# TABS
# =====================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🎤 Voice Input",
        "📚 RAG Context",
        "🤖 AI Response",
        "📊 Evaluation"
    ]
)

# =====================================
# TAB 1
# =====================================
with tab1:

    st.subheader("🎤 Speak your question")

    audio = audiorecorder("Click to record", "Recording...")

    if audio is not None and len(audio) > 0:

        # save audio
        audio.export("temp.wav", format="wav")

        st.audio("temp.wav")

        st.success("Audio recorded ✔️")


st.subheader("Ask a question")

mode = st.radio(
    "Choose input mode",
    ["🎤 Voice", "⌨️ Text"],
    horizontal=True
)
if mode == "⌨️ Text":

    question = st.text_input(
        "Type your question",
        placeholder="What is RAG?"
    )

    send = st.button("🚀 Ask")

    if send and question:
        with st.spinner("Searching knowledge base..."):
            
            answer, docs = ask_question(question)

        st.session_state["docs"] = docs
        st.session_state["answer"] = answer

        st.subheader("🤖 Response")
        st.success(answer)

    # ==========================
    # VOICE MODE
    # ==========================
else:

    audio = audiorecorder(
        "Click to record",
        "Stop recording"
    )

    if audio is not None and len(audio) > 0:

        audio.export("temp.wav", format="wav")

        st.audio("temp.wav")

        with st.spinner("Loading Whisper..."):
            model = WhisperModel(
                "small",
                device="cpu",
                compute_type="int8"
            )

        with st.spinner("Transcribing..."):
            segments, info = model.transcribe(
                "temp.wav",
                language="en",
                beam_size=5
            )

        question = " ".join(seg.text for seg in segments)

        st.subheader("📝 Transcription")
        st.text_area("", question, height=100)

# ==========================
# RAG
# ==========================

    # =====================================
    # WHISPER
    # =====================================

        with st.spinner("Loading Whisper..."):

            model = WhisperModel(
                "small",
                device="cpu",
                compute_type="int8"
            )

        with st.spinner("Transcribing..."):
            audio_path = "temp.wav"
            segments, info = model.transcribe(
                audio_path,
                language="en",
                beam_size=5
            )               

            transcription = " ".join([seg.text for seg in segments])

        st.subheader("📝 Transcription")

        st.text_area("", transcription, height=150)

        st.info(
            f"Language: {info.language} | Confidence: {info.language_probability:.2f}"
        )

        # =====================================
        # RAG
        # =====================================

        if transcription.strip():

            with st.spinner("Searching knowledge base..."):
                
                answer, docs = ask_question(transcription)

            st.session_state["docs"] = docs
            st.session_state["answer"] = answer
    
          
            # history (IMPORTANT)
            if "history" not in st.session_state:
                st.session_state["history"] = []

            st.session_state["history"].append({
                "q": transcription,
                "a": answer
            })

            st.success("RAG completed ✔️")

        else:
            st.error("No speech detected")


        if question:
            with st.spinner("Searching knowledge base..."):
                answer = ask_question(question)
                st.subheader("🤖 Response")
                st.write(answer)

# =====================================
# TAB 2
# =====================================

with tab2:

    st.subheader("Retrieved Documents")

    if "docs" in st.session_state:

        docs = st.session_state["docs"]

        for i, doc in enumerate(docs):

            st.markdown(
                f"### Document {i+1}"
            )

            st.write(
                doc.page_content[:1500]
            )

            st.divider()

    else:

        st.info(
            "Upload an audio file first."
        )

# =====================================
# TAB 3
# =====================================

# =====================================
# TAB 3
# =====================================

with tab3:

    st.subheader("🤖 AI Response")

    if "answer" in st.session_state:

        answer = st.session_state["answer"]

        st.success("Response generated")

        st.success(answer)

        try:
            import base64

            tts = gTTS(
                text=answer,
                lang="en"
            )

            tts.save("response.mp3")

            st.subheader("🔊 Audio Response")

            with open("response.mp3", "rb") as f:
                audio_bytes = f.read()

            st.audio(audio_bytes, format="audio/mp3")

            b64 = base64.b64encode(audio_bytes).decode()

            audio_html = f"""
            <audio autoplay controls>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """

            st.markdown(
                audio_html,
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"TTS Error: {e}")

    else:
        st.info("No answer yet.")

    st.divider()

    st.subheader("📜 Conversation History")

    if "history" in st.session_state and st.session_state["history"]:

        for i, item in enumerate(reversed(st.session_state["history"])):

            with st.expander(
                f"Conversation {len(st.session_state['history']) - i}"
            ):

                st.markdown("**🎤 Question**")
                st.write(item["q"])

                st.markdown("**🤖 Answer**")
                st.write(item["a"])

    else:
        st.info("No history yet.")
# TAB 4
# =====================================

with tab4:

    st.subheader("Project Evaluation")

    st.metric(
        "Indexed PDFs",
        9
    )

    st.metric(
        "Indexed Pages",
        320
    )

    st.metric(
        "Chunks",
        496
    )

    st.markdown("### Architecture")

    st.code("""
Audio
 ↓
Whisper
 ↓
Transcription
 ↓
ChromaDB
 ↓
Retriever
 ↓
Phi-3
 ↓
gTTS
 ↓
Audio Response
""")

# =====================================
# FOOTER
# =====================================

st.divider()

st.caption(
    "AIMS Senegal | Applied Generative and Agentic AI"
)