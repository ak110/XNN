@echo off

path %~dp0..\..\x64\Release;%PATH%

echo ====== �w�K ======
XNN XNN.conf hidden_units=32 hidden_layers=10

echo ====== ���� ======
XNN XNN.conf task=pred

pause
