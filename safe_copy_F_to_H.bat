@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Conservative copy workflow for long paths: copy F:\<folder> -> H:\<folder>, keep source.

echo ==============================================
echo Long-path safe copy (NO source deletion)
echo ==============================================
echo.

set "SRC_ROOT=F:\experiments\"
set "DST_ROOT=H:\"
set "ENABLE_DRYRUN=0"
set "NO_OVERWRITE=1"
set "LOG_DIR=%~dp0robocopy_log"
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
>> "%REPORT%" echo.

call :pick_folder 1 FOLDER1
if errorlevel 1 exit /b 1

call :pick_folder 2 FOLDER2 OPTIONAL
if errorlevel 1 exit /b 1

if not "%FOLDER2%"=="" (
  if /I "%FOLDER1%"=="%FOLDER2%" (
    echo [ERROR] Folder 1 and Folder 2 are the same. Please choose two different folders.
    exit /b 1
  )
)

set /a TOTAL_COPY_STEPS=1
if not "%FOLDER2%"=="" set /a TOTAL_COPY_STEPS=2
set /a COPY_STEP=0

set /a COPY_STEP+=1
echo [PROGRESS] Copy step %COPY_STEP%/%TOTAL_COPY_STEPS%: %FOLDER1%
call :copy_one "%FOLDER1%"
if errorlevel 1 exit /b 1

if not "%FOLDER2%"=="" (
  set /a COPY_STEP+=1
  echo [PROGRESS] Copy step %COPY_STEP%/%TOTAL_COPY_STEPS%: %FOLDER2%
  call :copy_one "%FOLDER2%"
  if errorlevel 1 exit /b 1
) else (
  echo [INFO] Folder 2 skipped.
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
set "LOG=%LOG_DIR%\robocopy_%NAME::=_%_%DATE:/=-%_%TIME::=-%.log"
set "LOG=%LOG: =0%"

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
robocopy "%SRC%" "%DST%" /E /COPY:DAT /DCOPY:DAT /R:1 /W:1 /MT:16 /Z /NP /TEE /LOG+:%LOG% !ROBO_EXTRA!
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

:pick_folder
set "PICK_NO=%~1"
set "OUTVAR=%~2"
set "OPTIONAL=%~3"
set /a IDX=0

echo.
echo Available folders under %SRC_ROOT%
for /f "delims=" %%D in ('dir /ad /b "%SRC_ROOT%"') do (
  set /a IDX+=1
  set "OPT_!IDX!=%%D"
  echo   !IDX!. %%D
)

if !IDX! EQU 0 (
  echo [ERROR] No folders found under %SRC_ROOT%
  exit /b 1
)

echo.
echo You can input a number or type the exact folder name.
if /I "%OPTIONAL%"=="OPTIONAL" echo Leave empty to skip this folder.
set "PICK="
set /p PICK=Choose folder !PICK_NO! ^(1-!IDX!^) : 

if "!PICK!"=="" (
  if /I "%OPTIONAL%"=="OPTIONAL" (
    set "%OUTVAR%="
    echo [SKIP] Folder !PICK_NO! skipped.
    exit /b 0
  ) else (
    echo [ERROR] Empty input.
    exit /b 1
  )
)

set "NAME="
set "NONNUM="
for /f "delims=0123456789" %%A in ("!PICK!") do set "NONNUM=%%A"
if not defined NONNUM (
  if !PICK! GEQ 1 if !PICK! LEQ !IDX! set "NAME=!OPT_%PICK%!"
)

if not defined NAME set "NAME=!PICK!"

if not exist "%SRC_ROOT%!NAME!" (
  echo [ERROR] Source does not exist: %SRC_ROOT%!NAME!
  exit /b 1
)

set "%OUTVAR%=!NAME!"
echo [SELECTED] Folder !PICK_NO!: !NAME!
exit /b 0
