@echo off
REM This script uses the directory where the .bat file is located, not the current directory

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"
REM Remove trailing backslash
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

echo Using script directory: %SCRIPT_DIR%

REM Set colors to green text on black background
color 0A

echo.
echo  DISCRAPE BUILD SCRIPT
echo  ---------------------
echo.

REM STEP 1: Clean up existing directories
echo [STEP 1] Cleaning up previous build...

if exist "%SCRIPT_DIR%\build" (
    echo - Removing previous build directory...
    rmdir /S /Q "%SCRIPT_DIR%\build"
)

if exist "%SCRIPT_DIR%\.venv" (
    echo - Removing previous virtual environment...
    rmdir /S /Q "%SCRIPT_DIR%\.venv"
)

REM STEP 2: Create fresh build directory
echo [STEP 2] Creating fresh build directory...
mkdir "%SCRIPT_DIR%\build"

REM STEP 3: Create virtual environment
echo [STEP 3] Creating new virtual environment...
cd "%SCRIPT_DIR%"
python -m venv "%SCRIPT_DIR%\.venv"

REM STEP 4: Activate virtual environment
echo [STEP 4] Activating virtual environment...
call "%SCRIPT_DIR%\.venv\Scripts\activate"

REM STEP 5: Install dependencies
echo [STEP 5] Installing dependencies...
pip install gradio pandas requests cryptography Pillow

REM STEP 6: Copy application file
echo [STEP 6] Copying application file...
copy "%SCRIPT_DIR%\discrape.py" "%SCRIPT_DIR%\build\" /Y

REM STEP 7: Create data directory
if not exist "%USERPROFILE%\Downloads\discrape" (
    echo [STEP 7] Creating data directory...
    mkdir "%USERPROFILE%\Downloads\discrape"
)

echo.
echo BUILD COMPLETED SUCCESSFULLY!
echo Run run.bat to launch DiScrape.
echo.

pause