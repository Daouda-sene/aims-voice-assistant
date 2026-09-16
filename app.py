import os
import base64
import tempfile

import streamlit as st
from faster_whisper import WhisperModel
from gtts import gTTS

from src.pipeline import ask_question


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AIMS Senegal Voice Agent",
    page_icon="🎤",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "docs" not in st.session_state:
    st.session_state["docs"] = []

if "answer" not in st.session_state:
    st.session_state["answer"] = ""

if "history" not in st.session_state:
    st.session_state["history"] = []

if "transcription" not in st.session_state:
    st.session_state["transcription"] = ""


# ============================================================
# LOAD WHISPER
# ============================================================

@st.cache_resource
def load_whisper():

    model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8"
    )

    return model


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    .title {
        color: #7A1E0E;
        font-size: 40px;
        font-weight: bold;
    }

    .subtitle {
        color: #003366;
        font-size: 18px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

if os.path.exists("assets/aims_logo.png"):

    st.sidebar.image(
        "assets/aims_logo.png",
        width="stretch"
    )

st.sidebar.title("AIMS Voice Agent")

st.sidebar.markdown(
    """
    ### Technologies

    - 🎤 Faster Whisper
    - 📚 ChromaDB
    - 🔎 MiniLM
    - 🤖 Groq / Llama
    - 🔊 gTTS

    ### Project

    Local Voice Agent with RAG

    **GAAI-AIMS Final Project**
    """
)


# ============================================================
# HEADER
# ============================================================

col1, col2 = st.columns([1, 5])

with col1:

    if os.path.exists("assets/aims_logo.png"):

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
        '<p class="subtitle">Whisper + RAG + LLM + TTS</p>',
        unsafe_allow_html=True
    )


st.divider()


# ============================================================
# KPI
# ============================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric("PDFs", "9")
c2.metric("Pages", "320")
c3.metric("Chunks", "496")
c4.metric("LLM", "Groq")


st.divider()


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🎤 Voice Input",
        "📚 RAG Context",
        "🤖 AI Response",
        "📊 Evaluation"
    ]
)


# ============================================================
# TAB 1 — INPUT
# ============================================================

with tab1:

    st.subheader("🎤 Ask your question")

    mode = st.radio(
        "Choose input mode",
        ["🎤 Voice", "⌨️ Text"],
        horizontal=True
    )


    # ========================================================
    # TEXT MODE
    # ========================================================

    if mode == "⌨️ Text":

        question = st.text_input(
            "Type your question",
            placeholder="What is RAG?"
        )

        send = st.button(
            "🚀 Ask",
            type="primary"
        )

        if send:

            if not question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                with st.spinner(
                    "Searching knowledge base..."
                ):

                    try:

                        # ------------------------------------
                        # RAG
                        # ------------------------------------

                        answer, docs = ask_question(
                            question
                        )

                        # ------------------------------------
                        # SAVE RESULTS
                        # ------------------------------------

                        st.session_state["answer"] = answer
                        st.session_state["docs"] = docs
                        st.session_state["transcription"] = question

                        # ------------------------------------
                        # HISTORY
                        # ------------------------------------

                        st.session_state["history"].append(
                            {
                                "q": question,
                                "a": answer
                            }
                        )

                        # ------------------------------------
                        # SUCCESS
                        # ------------------------------------

                        st.success(
                            "RAG completed ✔️"
                        )

                    except Exception as e:

                        st.error(
                            f"Error during RAG: {e}"
                        )


        # ----------------------------------------------------
        # DISPLAY TEXT RESULT
        # ----------------------------------------------------

        if (
            st.session_state.get("answer")
            and not send
        ):

            st.subheader(
                "🤖 AI Response"
            )

            st.write(
                st.session_state["answer"]
            )


        elif (
            send
            and st.session_state.get("answer")
        ):

            st.subheader(
                "🤖 AI Response"
            )

            st.write(
                st.session_state["answer"]
            )


    # ========================================================
    # VOICE MODE
    # ========================================================

    else:

        st.write(
            "Click the microphone and ask your question."
        )

        audio = st.audio_input(
            "🎤 Record your question"
        )


        if audio is not None:

            # =================================================
            # SAVE AUDIO
            # =================================================

            temp_audio = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            )

            temp_audio.write(
                audio.getvalue()
            )

            temp_audio.close()

            audio_path = temp_audio.name


            st.audio(
                audio.getvalue(),
                format="audio/wav"
            )

            st.success(
                "Audio recorded ✔️"
            )


            # =================================================
            # WHISPER
            # =================================================

            with st.spinner(
                "Loading Whisper..."
            ):

                try:

                    model = load_whisper()

                except Exception as e:

                    st.error(
                        f"Whisper loading error: {e}"
                    )

                    st.stop()


            # =================================================
            # TRANSCRIPTION
            # =================================================

            with st.spinner(
                "Transcribing audio..."
            ):

                try:

                    segments, info = model.transcribe(
                        audio_path,
                        language="en",
                        beam_size=5
                    )

                    transcription = " ".join(
                        segment.text
                        for segment in segments
                    ).strip()

                except Exception as e:

                    st.error(
                        f"Transcription error: {e}"
                    )

                    transcription = ""


            # =================================================
            # SHOW TRANSCRIPTION
            # =================================================

            if transcription:

                st.session_state["transcription"] = transcription

                st.subheader(
                    "📝 Transcription"
                )

                st.text_area(
                    "Recognized question",
                    transcription,
                    height=120
                )

                st.info(
                    f"Detected language: "
                    f"{info.language} | "
                    f"Confidence: "
                    f"{info.language_probability:.2f}"
                )


                # =============================================
                # RAG
                # =============================================

                with st.spinner(
                    "Searching knowledge base..."
                ):

                    try:

                        answer, docs = ask_question(
                            transcription
                        )

                        # -------------------------------------
                        # SAVE RESULTS
                        # -------------------------------------

                        st.session_state["answer"] = answer
                        st.session_state["docs"] = docs


                        # -------------------------------------
                        # HISTORY
                        # -------------------------------------

                        st.session_state["history"].append(
                            {
                                "q": transcription,
                                "a": answer
                            }
                        )


                        st.success(
                            "RAG completed ✔️"
                        )


                        # -------------------------------------
                        # DISPLAY ANSWER
                        # -------------------------------------

                        st.subheader(
                            "🤖 AI Response"
                        )

                        st.write(
                            answer
                        )


                    except Exception as e:

                        st.error(
                            f"RAG error: {e}"
                        )


            else:

                st.error(
                    "No speech detected."
                )


            # =================================================
            # DELETE TEMP AUDIO
            # =================================================

            try:

                os.remove(
                    audio_path
                )

            except Exception:

                pass


# ============================================================
# TAB 2 — RAG CONTEXT
# ============================================================

with tab2:

    st.subheader(
        "📚 Retrieved Documents"
    )

    docs = st.session_state.get(
        "docs",
        []
    )


    if docs:

        for i, doc in enumerate(docs):

            st.markdown(
                f"### 📄 Document {i + 1}"
            )

            if hasattr(
                doc,
                "page_content"
            ):

                content = doc.page_content

            else:

                content = str(doc)


            st.write(
                content[:1500]
            )


            if hasattr(
                doc,
                "metadata"
            ):

                if doc.metadata:

                    st.caption(
                        f"Metadata: {doc.metadata}"
                    )


            st.divider()


    else:

        st.info(
            "No documents retrieved yet. "
            "Ask a question first."
        )


# ============================================================
# TAB 3 — AI RESPONSE
# ============================================================

with tab3:

    st.subheader(
        "🤖 AI Response"
    )

    answer = st.session_state.get(
        "answer",
        ""
    )


    if answer:

        st.success(
            "Response generated ✔️"
        )

        st.write(
            answer
        )


        # ====================================================
        # TEXT TO SPEECH
        # ====================================================

        st.subheader(
            "🔊 Audio Response"
        )

        try:

            tts_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp3"
            )

            tts_path = tts_file.name

            tts_file.close()


            tts = gTTS(
                text=answer,
                lang="en"
            )

            tts.save(
                tts_path
            )


            with open(
                tts_path,
                "rb"
            ) as f:

                audio_bytes = f.read()


            st.audio(
                audio_bytes,
                format="audio/mp3"
            )


            # Autoplay

            b64 = base64.b64encode(
                audio_bytes
            ).decode()


            audio_html = f"""
            <audio controls autoplay>
                <source
                    src="data:audio/mp3;base64,{b64}"
                    type="audio/mp3"
                >
            </audio>
            """


            st.markdown(
                audio_html,
                unsafe_allow_html=True
            )


            try:

                os.remove(
                    tts_path
                )

            except Exception:

                pass


        except Exception as e:

            st.error(
                f"TTS Error: {e}"
            )


    else:

        st.info(
            "No answer yet. Ask a question first."
        )


    # ========================================================
    # HISTORY
    # ========================================================

    st.divider()

    st.subheader(
        "📜 Conversation History"
    )


    history = st.session_state.get(
        "history",
        []
    )


    if history:

        for i, item in enumerate(
            reversed(history)
        ):

            conversation_number = (
                len(history) - i
            )


            with st.expander(
                f"Conversation {conversation_number}"
            ):

                st.markdown(
                    "**🎤 Question**"
                )

                st.write(
                    item["q"]
                )


                st.markdown(
                    "**🤖 Answer**"
                )

                st.write(
                    item["a"]
                )


    else:

        st.info(
            "No conversation history yet."
        )


# ============================================================
# TAB 4 — EVALUATION
# ============================================================

with tab4:

    st.subheader(
        "📊 Project Evaluation"
    )


    c1, c2, c3 = st.columns(3)


    c1.metric(
        "Indexed PDFs",
        9
    )


    c2.metric(
        "Indexed Pages",
        320
    )


    c3.metric(
        "Chunks",
        496
    )


    st.markdown(
        "### Architecture"
    )


    st.code(
        """
🎤 Audio
   ↓
Streamlit Audio Input
   ↓
Faster Whisper
   ↓
Transcription
   ↓
MiniLM Embeddings
   ↓
ChromaDB
   ↓
Retriever
   ↓
Groq LLM
   ↓
Answer
   ↓
gTTS
   ↓
🔊 Audio Response
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AIMS Senegal | Applied Generative and Agentic AI"
)
