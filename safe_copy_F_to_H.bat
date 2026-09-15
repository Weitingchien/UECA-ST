@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Conservative copy workflow for long paths: copy F:\<folder> -> H:\ep_split10_t1te1v1_u7_disjoint\<folder>, keep source.

echo ==============================================
echo Long-path safe copy (NO source deletion)
echo ==============================================
echo.

set "SRC_ROOT=F:\experiments\"
set "DST_ROOT=H:\ep_split10_t1te1v1_u7_disjoint\"
set "SRC_ENTER_DEFAULT=F:\experiments\ep_split10_t1te1v1_u7_disjoint"
set "ENABLE_DRYRUN=0"
set "NO_OVERWRITE=1"
set "LOG_DIR=%~dp0robocopy_log"
set "RUN_ID=%RANDOM%%RANDOM%"
set /a LOG_SEQ=0
set /a VERIFY_TOTAL=0
set /a VERIFY_PASS=0
set /a VERIFY_FAIL=0
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set "REPORT=%LOG_DIR%\copy_verify_report_%DATE:/=-%_%TIME::=-%.txt"
set "REPORT=%REPORT: =0%"

set "ROBO_EXTRA="
if "%NO_OVERWRITE%"=="1" (
  rem Skip files that already exist at destination (regardless of timestamp)
  set "ROBO_EXTRA=/XC /XN /XO"
)

echo Default source parent: %SRC_ROOT%
set "IN="
set /p IN=Press Enter to use [%SRC_ENTER_DEFAULT%], or input source parent path: 
if "%IN%"=="\" (
  echo [WARN] Single "\" is not a valid source parent here. Using Enter default.
  set "IN="
)
if "%IN%"=="" (
  set "SRC_ROOT=%SRC_ENTER_DEFAULT%"
) else (
  set "SRC_ROOT=%IN%"
)
if not "%SRC_ROOT:~-1%"=="\" set "SRC_ROOT=%SRC_ROOT%\"

echo Fixed target parent: %DST_ROOT%
if not "%DST_ROOT:~-1%"=="\" set "DST_ROOT=%DST_ROOT%\"

echo.
if not exist "%SRC_ROOT%" (
  echo [ERROR] Source parent does not exist: %SRC_ROOT%
  exit /b 1
)
if not exist "%DST_ROOT%" (
  echo [INFO] Target parent does not exist. Creating: %DST_ROOT%
  mkdir "%DST_ROOT%"
  if errorlevel 1 (
    echo [ERROR] Failed to create target parent: %DST_ROOT%
    exit /b 1
  )
)

if "%ENABLE_DRYRUN%"=="1" (
  echo Dry-run mode: ON
) else (
  echo Dry-run mode: OFF - copy starts immediately
)
if "%NO_OVERWRITE%"=="1" (
  echo Overwrite mode: OFF - existing files in H: are preserved
) else (
  echo Overwrite mode: ON - existing files in H: may be updated
)
echo.

> "%REPORT%" echo Folder verification report
>> "%REPORT%" echo Generated at %DATE% %TIME%
>> "%REPORT%" echo Source parent: %SRC_ROOT%
>> "%REPORT%" echo Target parent: %DST_ROOT%
>> "%REPORT%" echo.

set /a COUNT=0
echo Input folder names to copy from source parent: %SRC_ROOT%
echo Leave empty and press Enter to start copying.
:collect
set "NAME="
set /a NEXT_INDEX=COUNT+1
set /p NAME=Folder name #!NEXT_INDEX!: 
if "%NAME%"=="" goto begin_copy
if not exist "%SRC_ROOT%%NAME%" (
  echo [WARN] Not found under source root: %NAME%
  goto collect
)
if not exist "%SRC_ROOT%%NAME%\" (
  echo [WARN] This is not a folder, skipped: %NAME%
  goto collect
)
call :is_duplicate "%NAME%" ALREADY_ADDED
if /I "!ALREADY_ADDED!"=="1" (
  echo [WARN] Folder already added: %NAME%
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
  echo [PROGRESS] Copy step %%I/%COUNT%: !CURRENT!
  call :copy_one "!CURRENT!"
  if errorlevel 1 exit /b 1
)

echo.
echo ==============================================
echo Copy finished. Source folders are NOT deleted.
echo Verification report: %REPORT%
echo.
echo Verification summary
echo Total folders checked: %VERIFY_TOTAL%
echo PASS: %VERIFY_PASS%
echo FAIL: %VERIFY_FAIL%
echo.
type "%REPORT%"
echo ==============================================
exit /b 0

:copy_one
set "NAME=%~1"
set "SRC=%SRC_ROOT%%NAME%"
set "DST=%DST_ROOT%%NAME%"
set /a LOG_SEQ+=1
set "LOG=%LOG_DIR%\robocopy_%RUN_ID%_!LOG_SEQ!.log"

echo.
echo [CHECK] %SRC% ^> %DST%
echo [NOW COPYING] %NAME%
if not exist "%SRC%" (
  echo [ERROR] Source does not exist: %SRC%
  exit /b 1
)

if "%ENABLE_DRYRUN%"=="1" (
  echo [DRY-RUN] Listing planned operations...
  echo [DRY-RUN] This may take time for large folders.
  robocopy "%SRC%" "%DST%" /E /COPY:DAT /DCOPY:DAT /R:1 /W:1 /NP /NFL /NDL /L /NJH /NJS !ROBO_EXTRA!
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
robocopy "%SRC%" "%DST%" /E /COPY:DAT /DCOPY:DAT /R:1 /W:1 /MT:16 /Z /NP /TEE /LOG:"%LOG%" !ROBO_EXTRA!
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

set /a VERIFY_TOTAL+=1
set "VERIFY_RESULT=FAIL"
if "%SRC_COUNT%"=="%DST_COUNT%" (
  set "VERIFY_RESULT=PASS"
  set /a VERIFY_PASS+=1
) else (
  set /a VERIFY_FAIL+=1
)

echo [VERIFY] %NAME% source files: %SRC_COUNT%, target files: %DST_COUNT%, result: %VERIFY_RESULT%
>> "%REPORT%" echo [%NAME%] source=%SRC_COUNT% target=%DST_COUNT% result=%VERIFY_RESULT%

exit /b 0

:count_files
set "%~2=0"
for /f %%C in ('powershell -NoProfile -Command "(Get-ChildItem -LiteralPath ''%~1'' -File -Recurse -Force -ErrorAction SilentlyContinue ^| Measure-Object).Count"') do set "%~2=%%C"
exit /b 0

:is_duplicate
set "%~2=0"
for /L %%I in (1,1,%COUNT%) do (
  if /I "!ITEM_%%I!"=="%~1" set "%~2=1"
)
exit /b 0
