from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


class _FallbackPromptBert(torch.nn.Module):
    """Pickle-safe fallback for legacy checkpoints that reference __main__.prompt_bert."""

    def __init__(self, bert_path: str = "./bert-base-chinese"):
        super().__init__()
        from transformers import BertForMaskedLM, BertTokenizer  # pylint: disable=import-outside-toplevel

        self.bert = BertForMaskedLM.from_pretrained(bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.bert.resize_token_embeddings(len(self.tokenizer))

    def forward(self, x_bert, labels):
        output = self.bert(x_bert, labels=labels)
        return output.loss, output.logits

    def get_cls_embeddings(self, x_bert):
        outputs = self.bert.bert(x_bert, output_hidden_states=True, return_dict=True)
        return outputs.last_hidden_state[:, 0, :]


def _register_prompt_bert_fallback_in_main() -> bool:
    """Register a compatible prompt_bert class in __main__ for legacy torch.load pickles.

    Some checkpoints were saved via torch.save(model, ...) while model class lived in __main__,
    which makes unpickling from another entrypoint fail with:
    AttributeError: Can't get attribute 'prompt_bert' on <module '__main__'>.
    """
    import __main__  # pylint: disable=import-outside-toplevel

    if hasattr(__main__, "prompt_bert"):
        return True

    try:
        import transformers  # pylint: disable=import-outside-toplevel,unused-import
    except Exception:
        return False

    __main__.prompt_bert = _FallbackPromptBert
    return True


DEFAULT_TEST_WEIGHT_AVG_MODE = "best_val_task_models"
DEFAULT_PARAMETER_AVERAGING_METHOD = "simple_mean"


def get_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def suggested_cli_argument_name() -> str:
    return "--test_weight_avg_mode"


def suggested_cli_mode_value() -> str:
    return DEFAULT_TEST_WEIGHT_AVG_MODE


def get_best_val_metadata_paths(save_path: str | Path, fold: int) -> Dict[str, Path]:
    save_path = Path(save_path)
    return {
        "emo": save_path / f"fold{fold}_best_val_checkpoint_emo.txt",
        "cause": save_path / f"fold{fold}_best_val_checkpoint_cause.txt",
        "pair": save_path / f"fold{fold}_best_val_checkpoint_pair.txt",
    }

# 讀單一 txt，解析出 stage、iteration、validation loss、checkpoint path
def parse_best_val_checkpoint_metadata(metadata_path: str | Path) -> Dict[str, Optional[str]]:
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    result: Dict[str, Optional[str]] = {
        "stage": None,
        "iteration": None,
        "validation_loss": None,
        "checkpoint_path": None,
    }

    with metadata_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if line.startswith("Best validation checkpoint (stage: ") and line.endswith(")"):
                result["stage"] = line[len("Best validation checkpoint (stage: "):-1]
            elif line.startswith("Iteration: "):
                result["iteration"] = line.split(": ", 1)[1]
            elif line.startswith("Validation loss: "):
                result["validation_loss"] = line.split(": ", 1)[1]
            elif line.startswith("Checkpoint path: "):
                result["checkpoint_path"] = line.split(": ", 1)[1]

    if not result["checkpoint_path"]:
        raise ValueError(f"Checkpoint path not found in metadata file: {metadata_path}")

    return result


def resolve_checkpoint_path(
    checkpoint_path: str | Path,
    metadata_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_absolute() and checkpoint_path.exists():
        return checkpoint_path

    repo_root_path = Path(repo_root) if repo_root is not None else get_repo_root()
    repo_candidate = repo_root_path / checkpoint_path
    if repo_candidate.exists():
        return repo_candidate.resolve()

    if metadata_path is not None:
        metadata_candidate = Path(metadata_path).resolve().parent / checkpoint_path
        if metadata_candidate.exists():
            return metadata_candidate.resolve()

    raise FileNotFoundError(
        f"Unable to resolve checkpoint path: {checkpoint_path}"
    )


def resolve_checkpoints_from_metadata(
    metadata_paths: Sequence[str | Path],
    repo_root: str | Path | None = None,
) -> List[Path]:
    resolved_paths: List[Path] = []
    for metadata_path in metadata_paths:
        info = parse_best_val_checkpoint_metadata(metadata_path)
        resolved_paths.append(
            resolve_checkpoint_path(
                info["checkpoint_path"],
                metadata_path=metadata_path,
                repo_root=repo_root,
            )
        )
    return resolved_paths


def load_checkpoint_object(checkpoint_path: str | Path, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(checkpoint_path, map_location=map_location)
    except AttributeError as exc:
        # Legacy full-model checkpoint saved from training script executed as __main__.
        err_msg = str(exc)
        needs_prompt_bert = "prompt_bert" in err_msg and "__main__" in err_msg
        if not needs_prompt_bert:
            raise

        registered = _register_prompt_bert_fallback_in_main()
        if not registered:
            raise RuntimeError(
                "Failed to auto-register prompt_bert fallback class for checkpoint loading. "
                "Please run in an environment where transformers is available."
            ) from exc

        return torch.load(checkpoint_path, map_location=map_location)


def extract_state_dict(model_or_state_dict) -> OrderedDict:
    if isinstance(model_or_state_dict, (dict, OrderedDict)):
        if "state_dict" in model_or_state_dict and isinstance(model_or_state_dict["state_dict"], (dict, OrderedDict)):
            return OrderedDict(model_or_state_dict["state_dict"])
        return OrderedDict(model_or_state_dict)
    if hasattr(model_or_state_dict, "state_dict"):
        return OrderedDict(model_or_state_dict.state_dict())
    raise TypeError(f"Unsupported checkpoint object type: {type(model_or_state_dict)!r}")


def validate_state_dict_keys(state_dicts: Sequence[OrderedDict]) -> List[str]:
    # 若輸入的 state_dict 清單為空，直接拋出錯誤避免後續索引失敗
    if not state_dicts:
        raise ValueError("No state_dicts provided")

    # 以第一個 state_dict 的 key 順序作為基準 (後續所有模型都要完全一致)
    reference_keys = list(state_dicts[0].keys())
    # 從第二個 state_dict 開始逐一檢查 (index 從 1 開始)
    for index, state_dict in enumerate(state_dicts[1:], start=1):
        # 取出目前這個 state_dict 的 key list (含順序)
        current_keys = list(state_dict.keys())
        # 只要 key 名稱或順序任一不同，就視為不相容
        if current_keys != reference_keys:
            # 計算「基準有、目前缺少」的 key (僅用於錯誤訊息)
            missing = sorted(set(reference_keys) - set(current_keys))
            # 計算「目前多出、基準沒有」的 key (僅用於錯誤訊息)
            extra = sorted(set(current_keys) - set(reference_keys))
            # 拋出詳細錯誤: 指出是哪一個 state_dict 不匹配，並附上缺少/多出 key 的前 10 個
            raise ValueError(
                "State dict keys do not match. "
                f"Mismatch at index {index}. Missing={missing[:10]}, Extra={extra[:10]}"
            )
    # 全部檢查通過後，回傳可安全迭代的 key list 
    return reference_keys


def _normalize_weights(num_models: int, weights: Optional[Sequence[float]]) -> List[float]:
    if weights is None:
        return [1.0 / num_models] * num_models

    if len(weights) != num_models:
        raise ValueError(f"Expected {num_models} weights, got {len(weights)}")

    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Weight sum must be positive")

    return [float(weight) / total for weight in weights]


def average_state_dicts(
    # 輸入多個模型的 state_dict (通常是 emo/cause/pair 三個)
    state_dicts: Sequence[OrderedDict],
    # 可選的加權係數；若為 None，後續會自動使用等權重
    weights: Optional[Sequence[float]] = None,
) -> OrderedDict:
    # 先驗證所有 state_dict 的 key (名稱與順序) 完全一致，並取回可迭代的 key list
    keys = validate_state_dict_keys(state_dicts)
    # 將使用者提供的權重正規化成總和為 1; 未提供則轉為平均權重
    normalized_weights = _normalize_weights(len(state_dicts), weights)

    # 建立一個新的有序字典，用來存放每個 key 平均後的參數
    averaged = OrderedDict()
    # 逐一處理每個參數 key
    for key in keys:
        # 收集此 key 在所有模型中的 tensor (例如三個模型各一個)，對於目前這個 key，去三個模型裡各抓一次同名參數
        tensors = [state_dict[key] for state_dict in state_dicts]
        # 取第一個 tensor 作為型別/分支判斷的參考
        first_tensor = tensors[0]
        # 只有浮點 tensor 才做加權平均 (如權重、偏置)
        if torch.is_floating_point(first_tensor):
            # 初始化加權總和（第一項進來前為 None）
            weighted_sum = None
            # 逐一把每個模型的 tensor 乘上對應權重後加總
            for tensor, weight in zip(tensors, normalized_weights):
                # 先切斷計算圖並轉成 float32，再乘以權重，提升平均計算穩定性
                contribution = tensor.detach().to(dtype=torch.float32) * weight
                # 第一個 contribution 直接指定; 其餘依序累加
                weighted_sum = contribution if weighted_sum is None else weighted_sum + contribution
            # 把平均結果轉回原始 tensor 的 dtype，並寫入 averaged 對應 key
            averaged[key] = weighted_sum.to(dtype=first_tensor.dtype)
        else:
            # 非浮點 tensor (例如某些索引/計數 buffer) 不做平均，直接複製第一個模型的值
            averaged[key] = first_tensor.detach().clone()
    # 回傳完整的平均後 state_dict
    return averaged


def average_state_dicts_simple_mean(
    state_dicts: Sequence[OrderedDict],
) -> OrderedDict:
    """不含加權平均版本: 不接受權重，且所有 key 一律做等權平均

    非浮點/非複數 tensor 會先轉成 float32 做 mean，再轉回原始 dtype
    """
    keys = validate_state_dict_keys(state_dicts)

    averaged = OrderedDict()
    floating_key_summaries = []
    non_floating_keys = []
    non_floating_key_summaries = []
    for key in keys:
        # 對於目前這個 key，去三個模型裡各抓一次同名參數
        tensors = [state_dict[key].detach() for state_dict in state_dicts]
        first_tensor = tensors[0]
        # 把多個 tensor 在第 0 維疊起來，例如原本每個 tensor 形狀是 (A, B)，三個模型會變成 (3, A, B)
        stacked = torch.stack(tensors, dim=0)
        if torch.is_floating_point(first_tensor):
            floating_key_summaries.append(
                f"{key}(shape={tuple(first_tensor.shape)}, dtype={first_tensor.dtype})"
            )
            averaged[key] = stacked.mean(dim=0).to(dtype=first_tensor.dtype)
        # File "/root/UECA-lambda/NeST/model_weight_averaging.py", line 270, in average_state_dicts_simple_mean
        # averaged[key] = stacked.mean(dim=0)
        # RuntimeError: mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: Long
        else:
            non_floating_keys.append(f"{key}({first_tensor.dtype})")
            sample_values = first_tensor.detach().reshape(-1)[:5].cpu().tolist() # reshape(-1) 把 tensor 攤平成 1 維，取前 5 個值顯示
            non_floating_key_summaries.append(
                f"{key}(shape={tuple(first_tensor.shape)}, dtype={first_tensor.dtype}, sample={sample_values})"
            )
            # 對於非浮點/非複數 tensor，先轉成 float32 做 mean，再轉回原始 dtype，讓後續的 mean(dim=0) 能執行
            averaged[key] = stacked.to(dtype=torch.float32).mean(dim=0).round().to(dtype=first_tensor.dtype)
    floating_preview = ", ".join(floating_key_summaries[:10]) if floating_key_summaries else "無"
    non_floating_preview = ", ".join(non_floating_key_summaries[:10]) if non_floating_key_summaries else "無"
    print(
        "[Model Averaging] simple_mean key 摘要: "
        f"floating={len(floating_key_summaries)} 個，前 10 個: {floating_preview}; "
        f"non_floating={len(non_floating_keys)} 個，前 10 個: {non_floating_preview}"
    )
    return averaged


def build_averaged_model_from_checkpoint_paths(
    checkpoint_paths: Sequence[str | Path],
    weights: Optional[Sequence[float]] = None,
    map_location: str | torch.device = "cpu",
    averaging_method: str = DEFAULT_PARAMETER_AVERAGING_METHOD,
):
    if not checkpoint_paths:
        raise ValueError("checkpoint_paths must not be empty") 

    loaded_objects = [load_checkpoint_object(path, map_location=map_location) for path in checkpoint_paths]
    state_dicts = [extract_state_dict(obj) for obj in loaded_objects]
    if averaging_method == "simple_mean":
        if weights is not None:
            raise ValueError("weights must be None when averaging_method='simple_mean'")
        print(
            f"[Model Averaging] build_averaged_model_from_checkpoint_paths 使用 simple_mean 分支，"
            f"共 {len(state_dicts)} 個 checkpoint"
        )
        averaged_state_dict = average_state_dicts_simple_mean(state_dicts)
    elif averaging_method == "weighted":
        print(
            f"[Model Averaging] build_averaged_model_from_checkpoint_paths 使用 weighted 分支，"
            f"共 {len(state_dicts)} 個 checkpoint，weights={weights}"
        )
        averaged_state_dict = average_state_dicts(state_dicts, weights=weights)
    else:
        raise ValueError(
            f"Unsupported averaging_method: {averaging_method}. "
            "Expected 'simple_mean' or 'weighted'."
        )

    template_model = loaded_objects[0]
    if not hasattr(template_model, "load_state_dict"):
        raise TypeError(
            "Template checkpoint is not a model object. "
            "Current training script saves full model objects, so this utility expects full models."
        )

    template_model.load_state_dict(averaged_state_dict, strict=True)
    return template_model, averaged_state_dict

# 三個 metadata 檔，回傳平均後模型
def build_averaged_model_from_metadata_paths(
    metadata_paths: Sequence[str | Path],
    weights: Optional[Sequence[float]] = None,
    map_location: str | torch.device = "cpu",
    repo_root: str | Path | None = None,
    averaging_method: str = DEFAULT_PARAMETER_AVERAGING_METHOD,
):
    checkpoint_paths = resolve_checkpoints_from_metadata(metadata_paths, repo_root=repo_root)
    averaged_model, averaged_state_dict = build_averaged_model_from_checkpoint_paths(
        checkpoint_paths,
        weights=weights,
        map_location=map_location,
        averaging_method=averaging_method,
    )
    return averaged_model, averaged_state_dict, checkpoint_paths

# 傳入 save_path 與 fold 取得平均模型
def build_averaged_model_for_fold(
    save_path: str | Path,
    fold: int,
    weights: Optional[Sequence[float]] = None,
    map_location: str | torch.device = "cpu",
    repo_root: str | Path | None = None,
    averaging_method: str = DEFAULT_PARAMETER_AVERAGING_METHOD,
):
    # 依據實驗輸出路徑與 fold 編號，組出 emo/cause/pair 三個最佳模型的 metadata 檔案路徑
    metadata_dict = get_best_val_metadata_paths(save_path, fold)
    # 依固定順序整理 metadata 路徑，供後續解析與平均流程使用
    metadata_paths = [metadata_dict["emo"], metadata_dict["cause"], metadata_dict["pair"]]
    # 從 metadata 解析實際 checkpoint，並載入三個模型後執行權重平均
    averaged_model, averaged_state_dict, checkpoint_paths = build_averaged_model_from_metadata_paths(
        metadata_paths,  # 三個任務對應的 metadata 路徑
        weights=weights,  # 若提供則使用自訂加權
        map_location=map_location,  # 指定 checkpoint 載入到 CPU 或特定裝置
        repo_root=repo_root,  # 可選：指定解析相對路徑時的專案根目錄
        averaging_method=averaging_method,  # 參數平均策略；預設使用 simple_mean
    )
    # 回傳平均後模型、平均後參數，以及本次使用到的 metadata/checkpoint 路徑資訊
    return {
        "model": averaged_model,  # 可直接拿來推論的平均後模型物件
        "state_dict": averaged_state_dict,  # 平均後的參數字典 (便於另存或比對)
        "metadata_paths": metadata_paths,  # 本次讀取的 metadata 檔案路徑清單
        "checkpoint_paths": checkpoint_paths,  # metadata 解析後實際使用的 checkpoint 路徑清單
        "averaging_method": averaging_method,  # 本次使用的參數平均方法
    }

# 把平均後模型另存成新的 pth
def save_averaged_model(
    model,
    state_dict: OrderedDict,
    output_model_path: str | Path,
    output_state_dict_path: str | Path | None = None,
) -> Tuple[Path, Optional[Path]]:
    output_model_path = Path(output_model_path)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, output_model_path)

    saved_state_dict_path: Optional[Path] = None
    if output_state_dict_path is not None:
        saved_state_dict_path = Path(output_state_dict_path)
        saved_state_dict_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, saved_state_dict_path)

    return output_model_path, saved_state_dict_path