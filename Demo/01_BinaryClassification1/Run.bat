@echo off

path %~dp0..\..\x64\Release;%PATH%

echo ====== �w�K ======
XNN XNN.conf

echo ====== ���� ======
XNN XNN.conf task=pred

pause
