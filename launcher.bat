@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
color 0A
title Trading Tools - Launcher Finale

REM ===============================
REM CONFIG
REM ===============================
set "PYCFG=python_default.cmd"
set "PYTHON_CMD=python"
set "REQ=requirements.txt"

set BASE_LIBS=requests websockets websocket-client ccxt numpy tzdata
set OPT_LIBS=pandas matplotlib mplfinance plyer psutil

REM ===============================
REM CARICA PYTHON DEFAULT
REM ===============================
if exist "%PYCFG%" (
    set /p PYTHON_CMD=<"%PYCFG%"
)

call :CHECK_PYTHON
if errorlevel 1 call :AUTO_DEFAULT

REM ===============================
REM MENU PRINCIPALE
REM ===============================
:MENU
cls
echo ==============================================
echo        TRADING TOOLS - MENU PRINCIPALE
echo ==============================================
echo.
echo Python attivo: %PYTHON_CMD%
echo.
echo 1  - Avvia script Python
echo B  - Installa librerie BASE
echo O  - Installa librerie OPZIONALI
echo A  - Installa TUTTO
echo R  - Installa requirements.txt (pip -r)
echo P  - Seleziona Python (salva default)
echo L  - Lista Python installati
echo H  - Apri Liquidation Heatmap (Coinglass)
echo Q  - Esci
echo.
set /p "SCELTA=>>> "

if /I "%SCELTA%"=="1" goto RUN_SCRIPTS
if /I "%SCELTA%"=="B" goto INSTALL_BASE
if /I "%SCELTA%"=="O" goto INSTALL_OPT
if /I "%SCELTA%"=="A" goto INSTALL_ALL
if /I "%SCELTA%"=="R" goto INSTALL_REQFILE
if /I "%SCELTA%"=="P" goto SELECT_PYTHON
if /I "%SCELTA%"=="L" goto LIST_PYTHON
if /I "%SCELTA%"=="H" goto OPEN_HEATMAP
if /I "%SCELTA%"=="Q" goto END

goto MENU

REM ===============================
REM AVVIO SCRIPT
REM ===============================
:RUN_SCRIPTS
cls
echo =============================
echo   SCRIPT PYTHON DISPONIBILI
echo =============================
echo.

set "n=0"
for %%f in (*.py) do (
    set /a n+=1
    set "FILE[!n!]=%%f"
)

if !n! EQU 0 (
    echo Nessun file .py trovato.
    goto RETURN_MENU
)

for /L %%i in (1,1,!n!) do (
    echo %%i^) !FILE[%%i]!
)

echo.
echo M = Menu principale
echo.
set /p "NUM=Selezione >>> "

if /I "%NUM%"=="M" goto MENU

set "SCRIPT=!FILE[%NUM%]!"
if not defined SCRIPT (
    echo Numero non valido.
    goto RETURN_MENU
)

echo.
echo Avvio: %SCRIPT%
start "PYTHON" cmd /k %PYTHON_CMD% "%SCRIPT%"
goto RETURN_MENU

REM ===============================
REM INSTALLAZIONI LIBRERIE (MODULARI)
REM ===============================
:INSTALL_BASE
cls
echo =============================
echo   INSTALL LIBRERIE BASE
echo =============================
call :INSTALL_LIST %BASE_LIBS%
goto RETURN_MENU

:INSTALL_OPT
cls
echo =============================
echo   INSTALL LIBRERIE OPZIONALI
echo =============================
call :INSTALL_LIST %OPT_LIBS%
goto RETURN_MENU

:INSTALL_ALL
cls
echo =============================
echo   INSTALL TUTTE LE LIBRERIE
echo =============================
call :INSTALL_LIST %BASE_LIBS% %OPT_LIBS%
goto RETURN_MENU

:INSTALL_LIST
call :CHECK_PYTHON
if errorlevel 1 (
    echo Python non valido: %PYTHON_CMD%
    exit /b
)

echo.
echo Aggiorno pip...
%PYTHON_CMD% -m pip install --upgrade pip

for %%L in (%*) do (
    echo.
    echo Installo %%L ...
    %PYTHON_CMD% -m pip install %%L
)
exit /b

REM ===============================
REM INSTALL REQUIREMENTS.TXT
REM ===============================
:INSTALL_REQFILE
cls
echo =============================
echo   INSTALL REQUIREMENTS.TXT
echo =============================
echo.

call :CHECK_PYTHON
if errorlevel 1 (
    echo Python non valido: %PYTHON_CMD%
    goto RETURN_MENU
)

if not exist "%REQ%" (
    echo [ERRORE] "%REQ%" non trovato nella cartella.
    goto RETURN_MENU
)

echo Uso Python: %PYTHON_CMD%
echo.
echo Aggiorno pip...
%PYTHON_CMD% -m pip install --upgrade pip
echo.
echo Installo dipendenze da "%REQ%"...
%PYTHON_CMD% -m pip install -r "%REQ%"

goto RETURN_MENU

REM ===============================
REM COINGLASS HEATMAP
REM ===============================
:OPEN_HEATMAP
cls
echo =============================
echo   COINGLASS - LIQUIDATION HEATMAP
echo =============================
echo.
echo Apro la pagina nel browser...
start "" "https://www.coinglass.com/liquidation-levels"
goto RETURN_MENU

REM ===============================
REM PYTHON
REM ===============================
:SELECT_PYTHON
cls
echo =============================
echo   SELEZIONE PYTHON
echo =============================
echo.
echo Versioni disponibili:
where py >nul 2>&1
if not errorlevel 1 (
    py -0p
) else (
    where python
)

echo.
echo Scrivi comando Python (es: py -3.11 oppure python)
echo M = Menu principale
echo.
set /p "CMD=>>> "

if /I "%CMD%"=="M" goto MENU
if "%CMD%"=="" goto MENU

%CMD% --version >nul 2>&1
if errorlevel 1 (
    echo Comando non valido.
    goto RETURN_MENU
)

echo %CMD%> "%PYCFG%"
set "PYTHON_CMD=%CMD%"
echo Salvato come default.
goto RETURN_MENU

:LIST_PYTHON
cls
echo =============================
echo   PYTHON INSTALLATI
echo =============================
where py >nul 2>&1
if not errorlevel 1 (
    py -0p
) else (
    where python
)
goto RETURN_MENU

REM ===============================
REM UTILITA'
REM ===============================
:RETURN_MENU
echo.
echo ----------------------------------------------
echo Premi INVIO per tornare al Menu principale (M)
echo ----------------------------------------------
pause >nul
goto MENU

:AUTO_DEFAULT
where py >nul 2>&1
if not errorlevel 1 (
    py -3.11 --version >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_CMD=py -3.11"
        exit /b
    )
)
set "PYTHON_CMD=python"
exit /b

:CHECK_PYTHON
%PYTHON_CMD% --version >nul 2>&1
if errorlevel 1 exit /b 1
exit /b 0

:END
cls
echo Uscita dal launcher.
pause
exit /b
