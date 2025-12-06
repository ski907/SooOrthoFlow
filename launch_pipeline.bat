@echo off
REM --- Try soo_images first, then sooorthoflow ---
call conda.bat activate soo_images 2>nul || call conda.bat activate sooorthoflow

REM --- Run the GUI ---
python ./pipeline/pipeline_gui.py