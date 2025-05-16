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
echo  DISCRAPE RUN SCRIPT
echo  ------------------
echo.

REM STEP 1: Verify the build directory and application file
if not exist "%SCRIPT_DIR%\build" (
    echo Error: Build directory not found. Please run simple_build.bat first.
    goto :end
)

if not exist "%SCRIPT_DIR%\build\discrape.py" (
    echo Error: discrape.py not found in build directory. Please run simple_build.bat first.
    goto :end
)

if not exist "%SCRIPT_DIR%\.venv" (
    echo Error: Virtual environment not found. Please run simple_build.bat first.
    goto :end
)

REM STEP 2: Activate the virtual environment
echo [STEP 1] Activating virtual environment...
call "%SCRIPT_DIR%\.venv\Scripts\activate"

REM STEP 3: Change to the build directory and run the application
echo [STEP 2] Running DiScrape...
cd "%SCRIPT_DIR%\build"

echo.
echo APPLICATION OUTPUT:
echo -----------------
echo.

python discrape.py

:end
pause
