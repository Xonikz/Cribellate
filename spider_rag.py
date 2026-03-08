import streamlit as st
import os
import tempfile
import base64
import fitz  # PyMuPDF
import whisper # The Audio AI
import gc
import json 
import re 
import time

# --- CORE RAG LIBRARIES ---
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma 
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

# --- DOCUMENT PROCESSING ---
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(page_title="Weave", page_icon="🕸️", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0d131a; color: #e0e6ed; }
    h1, h2, h3 { color: #38bdf8; font-family: 'Courier New', Courier, monospace; }
    [data-testid="stSidebar"] { background-color: #070b10; border-right: 2px solid #ea580c; }
    .stChatMessage { background-color: #17212b; border: 1px solid #1e293b; border-radius: 5px; }
    [data-testid="chatAvatarIcon-user"] { background-color: #ea580c; }
    [data-testid="chatAvatarIcon-assistant"] { background-color: #38bdf8; }
    .streamlit-expanderHeader { color: #ea580c; font-family: 'Courier New', Courier, monospace; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# WORKSPACE MANAGEMENT LOGIC
# ==========================================
BASE_WORKSPACE_DIR = "./weave_workspaces"

if not os.path.exists(BASE_WORKSPACE_DIR):
    os.makedirs(BASE_WORKSPACE_DIR)

def get_available_workspaces():
    workspaces = [f for f in os.listdir(BASE_WORKSPACE_DIR) if os.path.isdir(os.path.join(BASE_WORKSPACE_DIR, f))]
    if not workspaces:
        default_path = os.path.join(BASE_WORKSPACE_DIR, "Default_Project")
        os.makedirs(default_path)
        return ["Default_Project"]
    return workspaces

def load_workspace_settings(workspace_dir):
    settings_path = os.path.join(workspace_dir, "tts_settings.json")
    if os.path.exists(settings_path):
        with open(settings_path, "r") as f:
            return json.load(f)
    return {"emo_alpha": 1.0, "emo_text": "", "use_random": False, "enable_tts": False, "max_tokens": 80}

def save_workspace_settings(workspace_dir, settings):
    settings_path = os.path.join(workspace_dir, "tts_settings.json")
    with open(settings_path, "w") as f:
        json.dump(settings, f)

# NEW: Persistent Chat History Functions
def load_chat_history(workspace_dir):
    history_path = os.path.join(workspace_dir, "chat_history.json")
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            return json.load(f)
    return []

def save_chat_history(workspace_dir, messages):
    history_path = os.path.join(workspace_dir, "chat_history.json")
    with open(history_path, "w") as f:
        json.dump(messages, f)

# ==========================================
# BACKEND LOGIC
# ==========================================
@st.cache_resource
def load_models():
    llm = ChatOpenAI(
        base_url="http://localhost:1234/v1", 
        api_key="lm-studio", 
        temperature=0.1, 
        max_tokens=4096, 
        streaming=True,  
        model="any" 
    )
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    whisper_model = whisper.load_model("base")
    
    return llm, embeddings_model, whisper_model

llm, embeddings_model, whisper_model = load_models()

def transcribe_image_with_vision(image_bytes):
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Transcribe exactly what is written on this page. If there is handwriting, transcribe it as accurately as possible. Output ONLY the transcribed text, with no introductory or concluding remarks. Do not describe the image."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    )
    response = llm.invoke([message])
    return response.content

def process_document(uploaded_file, current_db_dir, use_vision):
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    try:
        chunks = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)

        if file_extension in [".wav", ".mp3", ".amr", ".flac", ".m4a"]:
            st.toast(f"Audio Transcription initiated for {uploaded_file.name}...")
            result = whisper_model.transcribe(temp_file_path)
            full_text = result["text"]
            document = Document(page_content=full_text, metadata={"source": uploaded_file.name})
            chunks = text_splitter.split_documents([document])

        elif file_extension in [".jpg", ".jpeg", ".png"] or (file_extension == ".pdf" and use_vision):
            st.toast(f"Optical Processing initiated for {uploaded_file.name}...")
            full_text = ""
            
            if file_extension in [".jpg", ".jpeg", ".png"]:
                with open(temp_file_path, "rb") as f:
                    full_text = transcribe_image_with_vision(f.read())
            else:
                doc = fitz.open(temp_file_path)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(dpi=150) 
                    img_bytes = pix.tobytes("jpeg")
                    page_text = transcribe_image_with_vision(img_bytes)
                    full_text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
                doc.close() 
            
            document = Document(page_content=full_text, metadata={"source": uploaded_file.name})
            chunks = text_splitter.split_documents([document])

        else:
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension in [".txt", ".md"]:
                loader = TextLoader(temp_file_path, encoding="utf-8")
            elif file_extension == ".docx":
                loader = Docx2txtLoader(temp_file_path)
            else:
                st.error(f"Unsupported file type: {file_extension}")
                return

            pages = loader.load()
            chunks = text_splitter.split_documents(pages)
        
        for chunk in chunks:
            chunk.metadata['source'] = uploaded_file.name

        if not os.listdir(current_db_dir): 
            Chroma.from_documents(chunks, embeddings_model, persist_directory=current_db_dir)
        else:
            db = Chroma(persist_directory=current_db_dir, embedding_function=embeddings_model)
            db.add_documents(chunks)
            
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {e}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def get_prompt_template(question, current_db_dir, chat_history_list):
    if not os.path.exists(current_db_dir) or not os.listdir(current_db_dir):
        return None, []
        
    db = Chroma(persist_directory=current_db_dir, embedding_function=embeddings_model)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    context_docs = retriever.invoke(question)
    
    context_text = ""
    sources_used = [] 
    
    for doc in context_docs:
        source_name = os.path.basename(doc.metadata.get('source', 'Unknown_Source'))
        if source_name not in sources_used:
            sources_used.append(source_name)
        
        source_index = sources_used.index(source_name) + 1
        context_text += f"--- [SOURCE {source_index}: {source_name}] ---\n{doc.page_content}\n\n"
        
    # NEW: Build conversational context from the last 4 messages
    chat_context = ""
    recent_messages = chat_history_list[-4:] if len(chat_history_list) > 0 else []
    for m in recent_messages:
        role_label = "USER" if m["role"] == "user" else "SYSTEM"
        chat_context += f"{role_label}: {m['content']}\n"
    
    prompt = f"""
    SYSTEM INTEL DOSSIERS:
    ---------------------
    {context_text}
    ---------------------

    PREVIOUS CONVERSATION LOG:
    {chat_context}

    SYSTEM QUERY: {question}
    
    INSTRUCTIONS:
    Synthesize an answer using ONLY the provided dossiers and the context of the previous conversation. 
    Format your final response using plain text paragraphs. DO NOT use markdown, asterisks, underscores, or special characters.
    Cite sources inline using bracketed numbers (e.g., [1]).
    When you are ready to answer, begin exactly with the phrase "FINAL SYNTHESIS:".
    """
    
    return prompt, sources_used

def parse_reasoning(raw_text):
    thoughts = ""
    answer = raw_text
    
    raw_text = raw_text.replace("<think>\n", "").replace("<think>", "")
    raw_text = raw_text.replace("</think>\n", "").replace("</think>", "")
    
    if "FINAL SYNTHESIS:" in raw_text:
        parts = raw_text.rsplit("FINAL SYNTHESIS:", 1) 
        thoughts = parts[0].strip()
        
        if thoughts.startswith("Thinking Process:"):
            thoughts = thoughts.replace("Thinking Process:", "", 1).strip()
            
        answer = parts[1].strip()
        
    return thoughts, answer

def generate_voice_clone_audio(text_to_speak, voice_target_path, output_wav_path, emo_alpha=1.0, emo_text="", use_random=False, max_tokens=80):
    try:
        from indextts.infer_v2 import IndexTTS2
        import torch
        import gc
        
        tts = IndexTTS2(
            cfg_path="checkpoints/config.yaml", 
            model_dir="checkpoints"
        )
        
        tts.infer(
            spk_audio_prompt=voice_target_path, 
            text=text_to_speak,
            output_path=output_wav_path,
            emo_alpha=emo_alpha,
            emo_text=emo_text,
            max_text_tokens_per_segment=max_tokens
        )
        
        del tts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
            
        return output_wav_path
        
    except ImportError as e:
        st.error(f"Import Error: {e}")
        return None
    except Exception as e:
        import traceback
        st.error(f"Voice generation failed: {e}")
        st.error(traceback.format_exc())
        return None

# ==========================================
# USER INTERFACE
# ==========================================
st.title("🕸️ Weave")
st.caption("Local Knowledge Retrieval & Synthesis Engine")

with st.sidebar:
    st.header("🗂️ Workspaces")
    available_workspaces = get_available_workspaces()
    selected_workspace = st.selectbox("Active Project:", available_workspaces)
    current_db_dir = os.path.join(BASE_WORKSPACE_DIR, selected_workspace)
    
    with st.expander("Create New Workspace"):
        new_workspace_name = st.text_input("Project Name (No spaces):")
        if st.button("Create"):
            if new_workspace_name:
                safe_name = new_workspace_name.replace(" ", "_")
                new_path = os.path.join(BASE_WORKSPACE_DIR, safe_name)
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                    st.success(f"Created {safe_name}!")
                    st.rerun() 
                else:
                    st.warning("Workspace already exists.")

    st.divider()
    
    st.header("⚙️ Ingestion Bay")
    st.write(f"Uploading to: **{selected_workspace}**")
    
    use_vision = st.checkbox("👁️ Use Optical Processing for PDFs")
    
    uploaded_files = st.file_uploader(
        "Select Documents", 
        type=["pdf", "txt", "md", "docx", "jpg", "jpeg", "png", "wav", "mp3", "amr", "flac", "m4a"], 
        accept_multiple_files=True
    )
    
    if st.button("Process & Index Dossiers"):
        if uploaded_files:
            with st.spinner(f"Indexing into {selected_workspace}..."):
                for file in uploaded_files:
                    process_document(file, current_db_dir, use_vision)
            st.success("Dossiers integrated successfully.")
        else:
            st.warning("Please upload a dossier first.")

    st.divider()
    
    st.subheader("🗣️ Voice Synthesis")
    
    if "ws_settings" not in st.session_state or st.session_state.get("current_workspace") != selected_workspace:
        st.session_state.ws_settings = load_workspace_settings(current_db_dir)
        
    enable_tts = st.checkbox("Enable Local Voice Clone Controls", value=st.session_state.ws_settings.get("enable_tts", False))
    
    saved_voice_path = os.path.join(current_db_dir, "target_voice.wav")
    has_saved_voice = os.path.exists(saved_voice_path)
    
    uploaded_voice = None
    emo_alpha = 1.0
    emo_text = ""
    use_random = False
    max_tokens = 80
    
    if enable_tts:
        if has_saved_voice:
            st.success("✅ Using saved workspace voice.")
            uploaded_voice = st.file_uploader("Replace Workspace Voice (5-10s .wav)", type=["wav"])
        else:
            uploaded_voice = st.file_uploader("Upload Voice Target (5-10s .wav)", type=["wav"])
            
        with st.expander("Voice Emotion & Speed Settings", expanded=True):
            emo_alpha = st.slider("Emotion Intensity (emo_alpha)", 0.0, 1.0, st.session_state.ws_settings.get("emo_alpha", 1.0))
            emo_text = st.text_input("Emotion Description (e.g. 'angry and fast')", st.session_state.ws_settings.get("emo_text", ""))
            use_random = st.checkbox("Add Randomness (use_random)", value=st.session_state.ws_settings.get("use_random", False))
            max_tokens = st.slider("Max Tokens/Segment (Lower = Faster)", 20, 200, st.session_state.ws_settings.get("max_tokens", 80))
            
            if st.button("💾 Save Voice & Settings to Workspace"):
                if uploaded_voice is not None:
                    with open(saved_voice_path, "wb") as f:
                        f.write(uploaded_voice.getbuffer())
                
                new_settings = {
                    "enable_tts": enable_tts,
                    "emo_alpha": emo_alpha,
                    "emo_text": emo_text,
                    "use_random": use_random,
                    "max_tokens": max_tokens
                }
                save_workspace_settings(current_db_dir, new_settings)
                st.session_state.ws_settings = new_settings
                st.success("Voice and settings saved to workspace!")
                st.rerun()

    st.divider()
    st.subheader("💾 Export Synthesis")
    if "messages" in st.session_state and len(st.session_state.messages) > 0:
        # Export the entire conversation log as a markdown file
        full_log = ""
        for m in st.session_state.messages:
            full_log += f"**{m['role'].upper()}**:\n{m['content']}\n\n"
            
        st.download_button(
            label="Download Project Chat Log (.md)",
            data=full_log,
            file_name=f"{selected_workspace}_log.md",
            mime="text/markdown"
        )
    else:
        st.write("No active synthesis to export.")

# --- CHAT INTERFACE & STATE MANAGEMENT ---
# Detect workspace switch and load history
if "current_workspace" not in st.session_state or st.session_state.current_workspace != selected_workspace:
    st.session_state.current_workspace = selected_workspace
    loaded_history = load_chat_history(current_db_dir)
    
    if not loaded_history:
        st.session_state.messages = [{"role": "assistant", "thoughts": "", "content": f"Connected to {selected_workspace}. Awaiting query...", "sources_list": []}]
    else:
        st.session_state.messages = loaded_history

for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message.get("thoughts"):
            with st.expander("⚙️ View System Logic"):
                st.markdown(message['thoughts'])
        st.markdown(message["content"])
        
        if message.get("sources_list"):
            st.markdown("---")
            st.caption("**References:**")
            for i, src in enumerate(message["sources_list"]):
                st.caption(f"[{i+1}] {src}")
                
        # Handle the audio logic for assistant messages
        if message["role"] == "assistant":
            # If audio already exists for this message, play it
            if message.get("audio") and os.path.exists(message["audio"]):
                st.audio(message["audio"], format="audio/wav")
            # If no audio exists, and TTS is enabled, show the button
            elif enable_tts and has_saved_voice:
                if st.button("🔊 Generate Audio", key=f"tts_btn_{index}"):
                    with st.spinner("Synthesizing audio on demand..."):
                        clean_tts_text = re.sub(r'\[\d+\]', '', message["content"])
                        clean_tts_text = re.sub(r'[*_<>#]', '', clean_tts_text)
                        
                        # Save the audio specifically tied to this message's index in the workspace folder
                        dedicated_audio_path = os.path.join(current_db_dir, f"audio_msg_{index}_{int(time.time())}.wav")
                        
                        output_wav_path = generate_voice_clone_audio(
                            text_to_speak=clean_tts_text, 
                            voice_target_path=saved_voice_path,
                            output_wav_path=dedicated_audio_path,
                            emo_alpha=emo_alpha,
                            emo_text=emo_text,
                            use_random=use_random,
                            max_tokens=max_tokens
                        )
                        if output_wav_path and os.path.exists(output_wav_path):
                            st.session_state.messages[index]["audio"] = output_wav_path
                            save_chat_history(current_db_dir, st.session_state.messages)
                            st.rerun()

if prompt := st.chat_input("Input Query..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "thoughts": "", "content": prompt})
    save_chat_history(current_db_dir, st.session_state.messages)

    with st.chat_message("assistant"):
        prompt_template, sources_list = get_prompt_template(prompt, current_db_dir, st.session_state.messages)
        
        if not prompt_template:
            st.error(f"ERROR: No dossiers detected in {selected_workspace}. Please upload documents.")
        else:
            message_placeholder = st.empty()
            full_raw_response = ""
            
            for chunk in llm.stream(prompt_template):
                full_raw_response += chunk.content
                message_placeholder.markdown(full_raw_response + "▌")
            
            thoughts, final_answer = parse_reasoning(full_raw_response)
            message_placeholder.empty()
            
            if thoughts:
                with st.expander("⚙️ View System Logic"):
                    st.markdown(thoughts)
            
            display_answer = final_answer if final_answer else full_raw_response
            st.markdown(display_answer)
            
            if sources_list:
                st.markdown("---")
                st.caption("**References:**")
                for i, src in enumerate(sources_list):
                    st.caption(f"[{i+1}] {src}")
            
            # Save the text response to history immediately. Audio will be handled by the button.
            st.session_state.messages.append({
                "role": "assistant", 
                "thoughts": thoughts, 
                "content": display_answer,
                "sources_list": sources_list,
                "audio": None
            })
            save_chat_history(current_db_dir, st.session_state.messages)
            st.rerun() # Rerun to render the new button for this message