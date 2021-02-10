echo #==============================================#
echo # CDLEML Process TF2 Object Detection API
echo #==============================================#

echo INFO: EXECUTE SCRIPT IN TARGET BASE FOLDER, e.g. samples/starwars_reduced

:: Constant Definition
set PYTHONENV=tf24
set BASEPATH=.
set SCRIPTPREFIX=..\..\scripts-and-guides\scripts
set PROJECTNAME=star_wars_detection

:: Environment preparation
echo Activate environment %PYTHONENV%
call conda activate %PYTHONENV%

echo Compress workspace for export to the training server

python %SCRIPTPREFIX%\training\zip_tool.py ^
--items="prepared-records, pre-trained-models, annotations, *.py, *.md, *.bat, *.sh" ^
--out="tmp/%PROJECTNAME%.zip"