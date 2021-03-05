echo #==============================================#
echo # CDLEML Process TF2 Object Detection API
echo #==============================================#

set FILENAME=validation_files_reduced.txt
set DIRNAME=validation_for_inference

echo #======================================================#
echo # Get all files names from a folder and save the names without extension to a file
echo #======================================================# 
::call (for %a in (".\%DIRNAME%\*") do @echo %~na) >%FILENAME%

::echo off
for /r %%D in (".\%DIRNAME%\*") do echo %%~nxD >> %FILENAME%