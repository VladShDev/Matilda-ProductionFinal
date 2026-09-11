@echo off
rem Her interpreter for a test or a probe, from this folder, with the GPU HIDDEN from it
rem (never GPU work beside a live body; a test hands the GPU to her body itself).
rem   py.cmd tests\doctor_test1.py t1
rem   py.cmd -m measure.check
cd /d "%~dp0"
set CUDA_VISIBLE_DEVICES=-1
"C:\Users\tscen\Documents\antropic\practice\project Matilda\.venv\Scripts\python.exe" %*
