@echo off
title Weave Installer
echo ===================================================
echo       WEAVE - Automated Installation Setup
echo ===================================================
echo.

echo [1/5] Checking system for FFmpeg...
winget install ffmpeg --accept-source-agreements --accept-package-agreements >nul 2>&1

echo [2/5] Creating isolated Python Virtual Environment...
python -m venv venv
call venv\Scripts\activate

echo [3/5] Installing core AI dependencies (This may take 10-15 minutes)...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo [4/5] Downloading IndexTTS 2 Engine Code...
:: Bypassing the Git LFS bandwidth error automatically
set GIT_LFS_SKIP_SMUDGE=1
git clone https://github.com/index-tts/index-tts.git temp_indextts
move temp_indextts\indextts .\indextts
rmdir /s /q temp_indextts

echo [5/5] Downloading IndexTTS 2 Voice Engine Weights...
huggingface-cli download IndexTeam/IndexTTS-2 --local-dir checkpoints

echo.
echo ===================================================
echo INSTALLATION COMPLETE! 
echo You can now close this window and double-click 'start_weave.bat'
echo ===================================================
pause