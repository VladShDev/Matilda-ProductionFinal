@echo off
rem MATILDA --- she lives.  One command, from this folder, on her own interpreter, her eye on the GPU.
rem   run.cmd                      a life from nothing
rem   run.cmd --keep NAME.duckdb   continue a life
cd /d "%~dp0"
set CUDA_VISIBLE_DEVICES=
"C:\Users\tscen\Documents\antropic\practice\project Matilda\.venv\Scripts\python.exe" her.py %*
