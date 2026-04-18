#!/bin/bash
# ==========================================
# 自動移動實驗資料夾腳本 (Linux/WSL 版本)
# ==========================================

SOURCE_DIR="/mnt/c/Users/chien/Downloads"
TARGET_BASE="/mnt/d/GithubRepo/UECA_ST/ep_split10_home_t1te1v1_u7_disjoint"

# 取得今天和昨天的日期 (格式: YYYY_MM_DD)
TODAY=$(date +%Y_%m_%d)
YESTERDAY=$(date -d "yesterday" +%Y_%m_%d)

# 檢查參數
if [ -z "$1" ]; then
    echo "============================================"
    echo "自動搜尋昨天 ($YESTERDAY) 和今天 ($TODAY) 的實驗資料夾:"
    echo "============================================"
    echo ""
    
    # 搜尋昨天建立的 prompt_ECPE 資料夾
    echo "【昨天 $YESTERDAY】"
    FOUND_YESTERDAY=0
    for folder in "$SOURCE_DIR"/prompt_ECPE_few_shot_ST_${YESTERDAY}*; do
        if [ -d "$folder" ]; then
            FOLDER_NAME=$(basename "$folder")
            echo "  [$((FOUND_YESTERDAY+1))] $FOLDER_NAME"
            ((FOUND_YESTERDAY++))
        fi
    done
    if [ $FOUND_YESTERDAY -eq 0 ]; then
        echo "  (沒有找到昨天的資料夾)"
    fi
    
    echo ""
    
    # 搜尋今天建立的 prompt_ECPE 資料夾
    echo "【今天 $TODAY】"
    FOUND_TODAY=0
    for folder in "$SOURCE_DIR"/prompt_ECPE_few_shot_ST_${TODAY}*; do
        if [ -d "$folder" ]; then
            FOLDER_NAME=$(basename "$folder")
            echo "  [$((FOUND_TODAY+1))] $FOLDER_NAME"
            ((FOUND_TODAY++))
        fi
    done
    if [ $FOUND_TODAY -eq 0 ]; then
        echo "  (沒有找到今天的資料夾)"
    fi
    
    echo ""
    echo "============================================"
    echo "用法: ./move_experiment.sh [資料夾名稱或萬用字元]"
    echo ""
    echo "範例:"
    echo "  ./move_experiment.sh prompt_ECPE_few_shot_ST_${TODAY}*       # 移動今天所有"
    echo "  ./move_experiment.sh prompt_ECPE_few_shot_ST_${YESTERDAY}*   # 移動昨天所有"
    echo "============================================"
    exit 0
fi

# 建立目標目錄（如果不存在）
mkdir -p "$TARGET_BASE"

# 搜尋並移動資料夾
FOUND=0
for folder in "$SOURCE_DIR"/$1; do
    if [ -d "$folder" ]; then
        FOLDER_NAME=$(basename "$folder")
        echo ""
        echo "找到資料夾: $FOLDER_NAME"
        
        # 檢查是否有巢狀資料夾 (內層有同名 prompt_ECPE 資料夾)
        # find "$folder": 在 $folder 資料夾內搜尋
        # -mindepth 1: 排除外層資料夾本身 (只搜尋子目錄)
        # -maxdepth 1: 只搜尋一層 (不進入更深的子目錄)
        # -type d: 只搜尋目錄 (d = directory)
        # -name "...": 名稱符合這個萬用字元 pattern
        # 2>/dev/null: 把錯誤訊息丟掉 (隱藏錯誤)
        # | head -1: 只取第一個結果
        INNER_FOLDER=$(find "$folder" -mindepth 1 -maxdepth 1 -type d -name "prompt_ECPE_few_shot_ST_*" 2>/dev/null | head -1)
        
        if [ -n "$INNER_FOLDER" ]; then
            # 有巢狀結構，直接移動內層資料夾的內容
            INNER_NAME=$(basename "$INNER_FOLDER")
            echo "[偵測到巢狀結構] 移動內層資料夾: $INNER_NAME"
            echo "正在移動至 $TARGET_BASE/$INNER_NAME/..."
            echo ""
            
            rsync -av --progress --remove-source-files "$INNER_FOLDER/" "$TARGET_BASE/$INNER_NAME/"
            MOVE_RESULT=$?
            
            if [ $MOVE_RESULT -eq 0 ]; then
                # 刪除來源空目錄 (包含外層和內層)
                find "$folder" -type d -empty -delete 2>/dev/null
                echo ""
                echo "[成功] 已移動: $INNER_NAME"
                ((FOUND++))
            else
                echo ""
                echo "[失敗] 無法移動: $INNER_NAME"
            fi
        else
            # 沒有巢狀，正常移動
            echo "正在移動至 $TARGET_BASE/$FOLDER_NAME/..."
            echo ""
            
            rsync -av --progress --remove-source-files "$folder/" "$TARGET_BASE/$FOLDER_NAME/"
            
            if [ $? -eq 0 ]; then
                find "$folder" -type d -empty -delete 2>/dev/null
                echo ""
                echo "[成功] 已移動: $FOLDER_NAME"
                ((FOUND++))
            else
                echo ""
                echo "[失敗] 無法移動: $FOLDER_NAME"
            fi
        fi
    fi
done

if [ $FOUND -eq 0 ]; then
    echo ""
    echo "沒有找到符合 \"$1\" 的資料夾"
    echo "請確認 Downloads 目錄中是否有此資料夾"
fi

echo ""
echo "完成! 共處理 $FOUND 個資料夾"
