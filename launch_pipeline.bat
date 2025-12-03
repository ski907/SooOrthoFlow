@echo off
REM --- Ensure conda is initialized ---
call conda.bat activate soo_images

REM --- Run the GUI ---
python ./pipeline/pipeline_gui.py
