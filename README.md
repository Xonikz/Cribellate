# Weave
A fully local, multi-modal RAG and TTS companion app for LM Studio

Weave is a fully local, one-click install RAG (Retrieval-Augmented Generation) companion application designed to run seamlessly alongside LM Studio. It provides an isolated workspace environment for ingesting diverse file types, chatting with your local LLMs, and generating high-fidelity audio synthesis.

Core Features:

One-Click Deployment: Automated batch scripts handle the complex installation of PyTorch CUDA, Whisper, and IndexTTS 2, completely bypassing manual dependency management.

Multi-Modal Ingestion: Process standard text and PDFs, transcribe audio notes using local Whisper, and utilize Vision LLMs to extract text from images and scanned documents. (Recommendation: Pair with a vision-capable model like Qwen3.5 9B in LM Studio for optimal optical processing).

Isolated Workspaces: Create dedicated project folders to organize reference documents. Each workspace maintains its own vector database and persistent chat history.

Transparent Reasoning: Expandable UI elements allow you to view the exact step-by-step logic your LLM used to synthesize its answers.

On-Demand Voice Cloning: Integrated IndexTTS 2 allows you to upload a short audio sample to clone a voice and generate high-quality .wav readouts of the AI's responses directly in the chat.

Export & Archive: Download full conversation logs as cleanly formatted Markdown files for external use.

Note: This is an initial build. Further refinements and code optimizations will be added in future updates.
