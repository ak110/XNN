@echo off

path %~dp0..\..\x64\Release;%PATH%

echo ====== ŠwK ======
XNN XNN.conf hidden_units=32 hidden_layers=10

echo ====== ŒŸØ ======
XNN XNN.conf task=pred

pause
