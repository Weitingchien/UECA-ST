# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Training Models

**Full training:**
```bash
python UECA_CE.py --savecheckpoint True --save_path prompt_ECPE/
```

**Training specific folds:**
```bash
python UECA_CE.py --savecheckpoint True --start_fold 1 --end_fold 3 --save_path prompt_ECPE/
```

**Few-shot training:**
```bash
python UECA_CE_few_shot.py --savecheckpoint True --save_path prompt_ECPE_few_shot/
```

**Few-shot self-training (pseudo-labeling):**
```bash
python UECA_CE_few_shot_ST.py --savecheckpoint True --save_path prompt_ECPE_few_shot_ST --training_iter 1
```

### Testing/Inference

**Standard testing:**
```bash
python UECA_CE.py --test_only True --checkpoint True --checkpointpath prompt_ECPE/ --batch_size 8 --dataset split10/
```

**Few-shot testing:**
```bash
python UECA_CE_few_shot.py --test_only True --checkpoint True --checkpointpath prompt_ECPE_few_shot/ --batch_size 8 --dataset split10_few_shot/
```

### Evaluation

**Standard evaluation:**
```bash
python eval_UECA_CE.py --gt_dir split10/ --pred_dir UECA-CE/
```

**Few-shot evaluation:**
```bash
python eval_UECA_CE_few_shot.py --gt_dir split10_few_shot/ --pred_dir UECA-CE_few_shot/
```

### Data Preparation

**Generate few-shot dataset (10% samples):**
```bash
python prepare_fewshot_data.py
```

## Architecture Overview

This is a Chinese emotion-cause pair extraction (ECPE) system using BERT-based masked language modeling:

### Core Components

1. **UECA_CE.py** - Main ECPE model for full supervised learning
2. **UECA_CE_few_shot.py** - Few-shot version using 10% labeled data
3. **UECA_CE_few_shot_ST.py** - Self-training variant with pseudo-labeling for unlabeled data

### Data Structure

- **split10/** - 10-fold cross-validation data splits
- **split10_few_shot/** - 10% sampled training data for few-shot learning
- **split10_few_shot_ST/** - Few-shot + unlabeled data for self-training
- **bert-base-chinese/**, **bert-tiny-chinese/** - Pre-trained Chinese BERT models

### Model Workflow

1. Documents are processed clause-by-clause with emotion-cause pair annotations
2. Text is formatted as "[CLS] 1 clause1 [MASK] [MASK] [MASK] [SEP] 2 clause2..."
3. MASK tokens are used for emotion ("是"/"非") and cause ("是"/"非") classification
4. Self-training uses confidence thresholds (default 0.9) for pseudo-labeling

### Key Parameters

- Batch size: 8
- Learning rate: 0.00001
- Training iterations: 20 (full), 3 (self-training)
- Max sequence length: 512 tokens
- Confidence threshold: 0.9 (self-training)

### Output Structure

Models save checkpoints as .pth files and generate prediction text files in respective result directories (UECA-CE/, UECA-CE_few_shot/).