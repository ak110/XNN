@echo off

path %~dp0..\..\x64\Release;%PATH%

echo ====== �w�K ======
XNN XNN.conf verbose=1 hidden_units=256 hidden_layers=5

echo ====== ���� ======
XNN XNN.conf task=pred

pause
