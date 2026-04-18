@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

REM ==========================================
REM 自動移動實驗資料夾腳本
REM 使用 robocopy 處理長路徑問題
REM ==========================================

REM 設定來源和目標目錄
set "SOURCE_DIR=C:\Users\chien\Downloads"
set "TARGET_BASE=D:\GithubRepo\UECA_ST"

REM 檢查參數
if "%~1"=="" (
    echo 用法: move_experiment.bat [資料夾名稱或萬用字元]
    echo.
    echo 範例:
    echo   move_experiment.bat prompt_ECPE_few_shot_ST_2026_01_18*
    echo   move_experiment.bat prompt_ECPE_few_shot_ST_2026_01_18_04_46_39_f1-10...
    echo.
    echo 腳本會自動:
    echo   1. 從 Downloads 找到符合的資料夾
    echo   2. 根據資料夾內容判斷目標 ep_xxx 資料夾
    echo   3. 使用 robocopy 移動 (支援長路徑)
    exit /b 1
)

REM 搜尋符合的資料夾
set "PATTERN=%~1"
set "FOUND=0"

for /d %%F in ("%SOURCE_DIR%\%PATTERN%") do (
    set "FOLDER_NAME=%%~nxF"
    echo.
    echo 找到資料夾: !FOLDER_NAME!
    
    REM 使用 robocopy 移動資料夾 (/E=包含子目錄, /MOVE=移動而非複製)
    REM /R:3=重試3次, /W:1=等待1秒
    echo 正在移動至 %TARGET_BASE%\ep_split10_home_t1te1v1_u7_disjoint\...
    
    robocopy "%%F" "%TARGET_BASE%\ep_split10_home_t1te1v1_u7_disjoint\!FOLDER_NAME!" /E /MOVE /R:3 /W:1 /NFL /NDL /NJH /NJS
    
    if !ERRORLEVEL! LEQ 7 (
        echo [成功] 已移動: !FOLDER_NAME!
        set /a FOUND+=1
    ) else (
        echo [失敗] 無法移動: !FOLDER_NAME!
    )
)

if %FOUND%==0 (
    echo.
    echo 沒有找到符合 "%PATTERN%" 的資料夾
    echo 請確認 Downloads 目錄中是否有此資料夾
)

echo.
echo 完成! 共處理 %FOUND% 個資料夾
pause
