::Use the special %0 variable to get the path to the current file.
::Write %~n0 to get just the filename without the extension.
::Write %~n0%~x0 to get the filename and extension.
::Also possible to write %~nx0 to get the filename and extension.

set filename=%~n0
echo Current filename without extension %filename%
echo Removed get_this from %filename:get_this_=%

