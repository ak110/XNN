@echo off

for /D %%i in (*) do (
    pushd "%%~i"
    echo ========================= %%~ni =========================
    echo. | call Run.bat
    echo.
    echo.
    popd
)

pause
