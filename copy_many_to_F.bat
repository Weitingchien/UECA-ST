@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Reusable conservative copy tool: copy multiple folders from SRC_PARENT to DST_PARENT.

echo ==============================================
echo Multi-folder copy helper (NO source deletion)
echo ==============================================
echo.

set "SRC_PARENT=D:\GithubRepo\UECA_ST\ep_split10_t1te1v1_u7_disjoint"
set "DST_PARENT=F:\experiments\ep_split10_t1te1v1_u7_disjoint"
set "ENABLE_DRYRUN=0"
set "LOG_DIR=%~dp0robocopy_log"

set /a TOTAL=0
set /a PASS=0
set /a FAIL=0
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set "REPORT=%LOG_DIR%\copy_many_report_%DATE:/=-%_%TIME::=-%.txt"
set "REPORT=%REPORT: =0%"

echo Default source parent: %SRC_PARENT%
set "IN="
set /p IN=Press Enter to keep, or input source parent path: 
if not "%IN%"=="" set "SRC_PARENT=%IN%"

echo Default target parent: %DST_PARENT%
set "IN="
set /p IN=Press Enter to keep, or input target parent path: 
if not "%IN%"=="" set "DST_PARENT=%IN%"

echo.
if not exist "%SRC_PARENT%" (
  echo [ERROR] Source parent does not exist: %SRC_PARENT%
  exit /b 1
)
if not exist "%DST_PARENT%" (
  echo [INFO] Target parent does not exist. Creating: %DST_PARENT%
  mkdir "%DST_PARENT%"
  if errorlevel 1 (
    echo [ERROR] Failed to create target parent: %DST_PARENT%
    exit /b 1
  )
)

if "%ENABLE_DRYRUN%"=="1" (
  echo Dry-run mode: ON
) else (
  echo Dry-run mode: OFF - copy starts immediately
)
echo.

> "%REPORT%" echo Copy verification report
>> "%REPORT%" echo Generated at %DATE% %TIME%
>> "%REPORT%" echo Source parent: %SRC_PARENT%
>> "%REPORT%" echo Target parent: %DST_PARENT%
>> "%REPORT%" echo.

set /a COUNT=0
echo Input folder names to copy from source parent.
echo Leave empty and press Enter to start copying.
:collect
set "NAME="
set /p NAME=Folder name #%COUNT%+1: 
if "%NAME%"=="" goto begin_copy
if not exist "%SRC_PARENT%\%NAME%" (
  echo [WARN] Not found under source parent: %NAME%
  goto collect
)
if not exist "%SRC_PARENT%\%NAME%\" (
  echo [WARN] This is not a folder, skipped: %NAME%
  goto collect
)
set /a COUNT+=1
set "ITEM_!COUNT!=%NAME%"
echo [ADD] %NAME%
goto collect

:begin_copy
if %COUNT% EQU 0 (
  echo [ERROR] No folders to copy.
  exit /b 1
)

echo.
echo Starting copy for %COUNT% folders...
for /L %%I in (1,1,%COUNT%) do (
  set "CURRENT=!ITEM_%%I!"
  call :copy_one "!CURRENT!"
  if errorlevel 1 exit /b 1
)

echo.
echo ==============================================
echo Copy finished. Source folders are NOT deleted.
echo Verification report: %REPORT%
echo.
echo Verification summary
echo Total folders checked: %TOTAL%
echo PASS: %PASS%
echo FAIL: %FAIL%
echo ==============================================
echo.
type "%REPORT%"
exit /b 0

:copy_one
set "NAME=%~1"
set "SRC=%SRC_PARENT%\%NAME%"
set "DST=%DST_PARENT%\%NAME%"
rem Keep log file name short to avoid command-line length/argument parsing issues.
set "LOG=%LOG_DIR%\robocopy_%DATE:/=-%_%TIME::=-%_%RANDOM%.log"
set "LOG=%LOG: =0%"

echo.
echo [CHECK] %SRC% ^> %DST%
if not exist "%SRC%" (
  echo [ERROR] Source does not exist: %SRC%
  exit /b 1
)

if "%ENABLE_DRYRUN%"=="1" (
  echo [DRY-RUN] Listing planned operations...
  robocopy "%SRC%" "%DST%" /E /COPY:DAT /DCOPY:DAT /R:1 /W:1 /NP /NFL /NDL /L /NJH /NJS
  set "RC=%ERRORLEVEL%"
  if !RC! GEQ 8 (
    echo [ERROR] Dry-run failed for %NAME% - robocopy code: !RC!
    exit /b !RC!
  )
  echo [DRY-RUN] Preview completed. No files were copied in this step.
) else (
  echo [DRY-RUN] Skipped.
)

echo [COPY] Start copying...
robocopy "%SRC%" "%DST%" /E /COPY:DAT /DCOPY:DAT /R:1 /W:1 /MT:16 /Z /NP /TEE /LOG+:"%LOG%"
set "RC=%ERRORLEVEL%"
if %RC% GEQ 8 (
  echo [ERROR] Copy failed for %NAME% - robocopy code: %RC%
  echo Check log: %LOG%
  exit /b %RC%
)

echo [OK] %NAME% copied. (robocopy code: %RC%)
echo Log: %LOG%

call :count_files "%SRC%" SRC_COUNT
call :count_files "%DST%" DST_COUNT

set /a TOTAL+=1
set "RESULT=FAIL"
if "%SRC_COUNT%"=="%DST_COUNT%" (
  set "RESULT=PASS"
  set /a PASS+=1
) else (
  set /a FAIL+=1
)

echo [VERIFY] %NAME% source files: %SRC_COUNT%, target files: %DST_COUNT%, result: %RESULT%
>> "%REPORT%" echo [%NAME%] source=%SRC_COUNT% target=%DST_COUNT% result=%RESULT%

exit /b 0

:count_files
set "%~2=0"
for /f %%C in ('powershell -NoProfile -Command "(Get-ChildItem -LiteralPath ''%~1'' -File -Recurse -Force -ErrorAction SilentlyContinue ^| Measure-Object).Count"') do set "%~2=%%C"
exit /b 0
