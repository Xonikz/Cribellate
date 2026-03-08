# 🕸️ Weave
**A fully local, multi-modal RAG and TTS companion app for LM Studio**

![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)

Weave is a fully offline, one-click install Retrieval-Augmented Generation (RAG) companion application designed to run seamlessly alongside **LM Studio**. It provides an isolated workspace environment for ingesting diverse file types, chatting with your local LLMs, and generating high-fidelity audio synthesis.


---<img width="720" height="480" alt="Weave_GUI" src="https://github.com/user-attachments/assets/e9a32b4f-4b45-4b01-b91a-4761216caa24" />

## ✨ Core Features

* **One-Click Deployment:** Automated batch scripts handle the complex installation of PyTorch CUDA, Whisper, and IndexTTS 2, completely bypassing manual dependency management.
* **Multi-Modal Ingestion:** Process standard text and PDFs, transcribe audio notes using local Whisper, and utilize Vision LLMs to extract text from images and scanned documents. 
  > *Recommendation: Pair with a vision-capable model like Qwen3.5-VL in LM Studio for optimal optical processing.*
* **Isolated Workspaces:** Create dedicated project folders to organize reference documents. Each workspace maintains its own Chroma vector database and persistent chat history.
* **Transparent Reasoning:** Expandable UI elements allow you to view the exact step-by-step logic your LLM used to synthesize its answers.
* **On-Demand Voice Cloning:** Integrated IndexTTS 2 allows you to upload a short audio sample to clone a voice and generate high-quality `.wav` readouts of the AI's responses directly in the chat.
* **Export & Archive:** Download full conversation logs as cleanly formatted Markdown files for external use.

---

## 🛠️ Prerequisites

Before installing Weave, ensure you have the following ready:
1. **Python 3.10 or higher** installed on your system (ensure it is added to your system PATH).
2. **LM Studio** installed and running.
3. **LM Studio Local Server:** You must start the local inference server in LM Studio (running on the default `http://localhost:1234/v1`).

---

## 🚀 Installation & Setup

Weave uses a "fetch-and-build" architecture to automatically isolate its environment and download the necessary AI weights without cluttering your system.

1. Clone or download this repository to your local machine.
2. Double-click the `install.bat` file. 
3. Allow the terminal to run. It will automatically check for FFmpeg, build a Python virtual environment, install the CUDA drivers, and download the IndexTTS 2 voice models. *(Note: This process downloads several gigabytes of AI models and may take 10-15 minutes depending on your internet connection).*

---

## 💻 Usage

1. Open **LM Studio**, load your preferred LLM (and Vision model if processing images), and click **Start Server**.
2. Double-click `Start_Weave.bat` in your Weave folder.
3. Your browser will automatically open the Weave UI. 
4. Create a new workspace, upload your reference files into the Ingestion Bay, and begin chatting with your localized data!

---

## 📝 Roadmap & Notes
This is an initial release. Further refinements, UI enhancements, and code optimizations will be added in future updates.

---
*Developed by Xonikz Independent Industries*
