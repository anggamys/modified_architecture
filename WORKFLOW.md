# Complete Workflow: Training → Inference → Pseudo-Labeling

This document explains the complete data flow and relationships between training and inference processes.

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     COMPLETE WORKFLOW                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [Training Data] → main.py → [Train Model + Outputs] → inference.py│
│                   ├── train.py │              ├── char_vocab       │
│                   │            │              ├── class_mappings   │
│                   │            │              ├── best_model       │
│                   │            │              └── ...              │
│                   └────────────┴──────────────────────────────────→ run_inference()
│                                                                     │
│                                          [Pseudo-labeled Data CSV] │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Stage 1: Training (main.py + train.py)

### Input
- **data_path**: CSV file dengan kolom `[token, pos_tag]`
- **model_name**: HuggingFace BERT model (default: indobenchmark/indobert-base-p1)
- **config_name**: Scenario name (M1-M6, M6a, M6b) - optional

### Process
1. Load data dan split train:val:test (70:15:15)
2. Build character vocabulary dari training tokens
3. Create HybridModel dengan BERT + char-level encoder + optional CRF
4. Train dengan early stopping pada validation metric
5. Evaluate best model pada test set
6. Save outputs untuk inference stage

### Output Files
**Location**: `outputs/{config_name}/` (atau `outputs/Custom/` jika config_name="")

```
outputs/M6/
├── best_model_m6.pt                    ← Model weights untuk inference
├── char_vocab_m6.json                  ← Char → ID mapping
│                                          Format: {"a": 1, "b": 2, ...}
│
├── class_mappings_m6.json              ← POS tag ↔ ID mapping
│                                          Format: {
│                                            "class_to_idx": {"NN": 0, "VB": 1, ...},
│                                            "idx_to_class": {"0": "NN", "1": "VB", ...}
│                                          }
│
├── classification_report_m6.json       ← Precision/Recall/F1 per tag
├── test_results_m6.csv                 ← Token-level predictions on test set
│                                          Columns: [sentence_id, token_idx, token, 
│                                                   true_label, pred_label, correct, ...]
│
├── test_results_m6.json                ← Same as above, structured format
└── crf_transitions_m6.csv              ← CRF learned transitions (if use_crf=True)
                                           Index/Columns: POS tags
                                           Values: transition scores
```

### Usage Example

```bash
# Train model M6 (default: IndoBERTweet + Char-BiLSTM + CRF)
python main.py \
  --data_path ./data/sample-anotasi-merge-valid.csv \
  --model_name "dafqi/IndoBertTweet" \
  --config M6 \
  --epochs 20 \
  --batch_size 16

# Or train with custom config (no --config flag)
python main.py \
  --data_path ./data/sample-anotasi-merge-valid.csv \
  --model_name "indobenchmark/indobert-base-p1" \
  --char_type bilstm \
  --use_crf true \
  --epochs 20
```

### Output Logging
At the end of training, logs show:
```
Semua file (Model, Vocab, Class Mapping) berhasil disimpan di folder: outputs/M6/
WORKFLOW: TRAINING → INFERENCE
1. Training selesai. Output tersimpan di: outputs/M6/
2. Untuk pseudo-labeling unlabeled corpus, jalankan inference.py:
   python inference.py --config_name M6 --corpora_dir ./raw_corpora --output ./data_pseudo_label.csv
...
```

---

## Stage 2: Inference (inference.py)

### Input

#### Option A: Using config_name (Recommended)
```bash
python inference.py --config_name M6 --corpora_dir ./raw_corpora
```
- Automatically loads from `outputs/M6/`
- Reads config from `config.yml` untuk char_type dan use_crf

#### Option B: Using explicit paths
```bash
python inference.py \
  --model_path outputs/M6/best_model_m6.pt \
  --vocab_path outputs/M6/char_vocab_m6.json \
  --mapping_path outputs/M6/class_mappings_m6.json \
  --corpora_dir ./raw_corpora
```

#### Option C: Default paths (legacy, for compatibility)
```bash
python inference.py --corpora_dir ./raw_corpora
```
- Expects `./best_model.pt`, `./char_vocab.json`, `./class_mappings.json`

### Process
1. Load corpus dari folder (auto-detects WhatsApp vs standard text format)
2. Load trained model, vocab, class mappings
3. For each batch of sentences:
   - Tokenize dengan BERT tokenizer
   - Prepare char-level features
   - Forward through model
   - Decode predictions → POS tag names
4. Save incremental CSV outputs

### Output Files
**Location**: Determined by `--output` flag (default: `./data_pseudo_label.csv`)

#### Format A: Single file (--split_by=none)
```
data_pseudo_label.csv
├── file_id: Source file name
├── sentence_id: Global sentence number (1-indexed)
├── token_index: Position in sentence (1-indexed)
├── token: The actual word/token
├── pos_tag_pred: Predicted POS tag (dari model)
├── pos_tag_koreksi: EMPTY - untuk manual correction
│
```

#### Format B: Per-source file (--split_by=file, default)
```
data_pseudo_label/
├── data_pseudo_label_file1.csv
├── data_pseudo_label_file2.csv
└── ...
```
Each file contains predictions for one source file.

#### Format C: By token count (--split_by=rows)
```
data_pseudo_label/
├── data_pseudo_label_part001.csv  (500,000 sentences)
├── data_pseudo_label_part002.csv  (500,000 sentences)
└── ...
```

### Key Output Columns
| Column | Format | Example | Notes |
|--------|--------|---------|-------|
| file_id | string | "chat_001" | From source filename |
| sentence_id | int | 1, 2, 3, ... | Global across all files |
| token_index | int | 1, 2, 3, ... | Per-sentence numbering |
| token | string | "Saya" | Original word |
| pos_tag_pred | string | "NN", "VB" | Model prediction |
| pos_tag_koreksi | string | "" | **Manual correction field** |

### Usage Examples

```bash
# Option 1: Load from training outputs (config_name)
python inference.py \
  --config_name M6 \
  --corpora_dir ./raw_corpora \
  --output ./data_pseudo_label.csv \
  --batch_size 32

# Option 2: Explicit paths
python inference.py \
  --model_path outputs/M6/best_model_m6.pt \
  --vocab_path outputs/M6/char_vocab_m6.json \
  --mapping_path outputs/M6/class_mappings_m6.json \
  --corpora_dir ./raw_corpora \
  --output ./data_pseudo_label.csv

# Option 3: Split by file (default)
python inference.py \
  --config_name M6 \
  --corpora_dir ./raw_corpora \
  --output ./data_pseudo_label.csv \
  --split_by file

# Option 4: Split by row count
python inference.py \
  --config_name M6 \
  --corpora_dir ./raw_corpora \
  --output ./data_pseudo_label.csv \
  --split_by rows \
  --rows_per_file 1000000

# Option 5: With custom BERT model
python inference.py \
  --config_name M6 \
  --corpora_dir ./raw_corpora \
  --model_name "dafqi/IndoBertTweet" \
  --output ./data_pseudo_label.csv
```

---

## Stage 3: Post-Processing & Integration

### Step 1: Manual Correction
```
# Edit the CSV files:
# For each row with incorrect pos_tag_pred, fill in pos_tag_koreksi with correct tag
# Leave pos_tag_koreksi empty if pos_tag_pred is correct

file_id, sentence_id, token_index, token,         pos_tag_pred, pos_tag_koreksi
chat_1,  1,           1,            Saya,          NN,           
chat_1,  1,           2,            makan,         NN,           VB          ← Corrected
chat_1,  1,           3,            nasi,          NN,           
```

### Step 2: Create Final Training Dataset
```python
import pandas as pd

# Read pseudo-labeled data
pseudo_df = pd.read_csv("data_pseudo_label.csv")

# Use corrected labels if available, else use predictions
pseudo_df["pos_tag_final"] = pseudo_df["pos_tag_koreksi"].fillna(pseudo_df["pos_tag_pred"])

# Combine with original training data
train_df = pd.read_csv("data/sample-anotasi-merge-valid.csv")
augmented_df = pd.concat([train_df, pseudo_df[["token", "pos_tag_final"]].rename(columns={"pos_tag_final": "pos_tag"})], ignore_index=True)

# Train again with augmented dataset
augmented_df.to_csv("data/augmented-training-data.csv", index=False)
```

---

## Data Flow & File Dependencies

### Training Phase
```
Input Data (CSV)
    ↓
    main.py calls:
    ├─ build_char_vocab()      → char_vocab_{config}.json
    ├─ create model config     → HybridModel instantiation
    ├─ train_model()
    │  └─ train.py functions:
    │     ├─ train_one_epoch() → update model weights
    │     └─ evaluate()        → validation loss & accuracy
    ├─ evaluate_with_tokens()  → collect test predictions
    ├─ save_test_results()     → test_results_{config}.{csv,json}
    ├─ compute_classification_report()
    │                          → classification_report_{config}.json
    ├─ save class_mappings     → class_mappings_{config}.json
    ├─ save model weights      → best_model_{config}.pt
    └─ save CRF transitions    → crf_transitions_{config}.csv (optional)
```

### Inference Phase
```
Unlabeled Corpus (txt files)
    ↓
    inference.py calls:
    ├─ load_model()
    │  ├─ load best_model_{config}.pt      ← Restore trained weights
    │  ├─ load char_vocab_{config}.json    ← Load char→id mapping
    │  └─ load class_mappings_{config}.json ← Load tag↔id mapping
    ├─ load_corpus_from_folder()           ← Process raw text
    ├─ run_inference()
    │  ├─ _predict_batch()                 → Get predictions from model
    │  ├─ _decode_preds()                  → Convert indices → tag names
    │  └─ _flush()                         → Write CSV incrementally
    └─ Output: data_pseudo_label.csv (or split files)
```

---

## Configuration Scenarios (config.yml)

Each scenario defines char_type, use_crf, and model_name:

| Scenario | Model | Char | CRF | Purpose |
|----------|-------|------|-----|---------|
| M1 | IndoBERT | None | No | Baseline |
| M2 | IndoBERT | None | Yes | + CRF syntax modeling |
| M3 | IndoBERT | None | Yes | + Word-BiLSTM |
| M4 | IndoBERT | CNN | Yes | + Char-CNN |
| **M5** | IndoBERT | BiLSTM | Yes | + Char-BiLSTM |
| **M6** | IndoBERTweet | BiLSTM | Yes | Domain adapted |
| M6a | IndoBERTweet | CNN | Yes | Control: BiLSTM vs CNN |
| M6b | IndoBERTweet | None | Yes | Control: Char impact |

When using `--config_name M6`:
1. Automatically loads from `outputs/M6/`
2. Reads char_type="bilstm" and use_crf=true from config.yml
3. Uses model_name="dafqi/IndoBertTweet" from config.yml

---

## Troubleshooting

### Issue: "Model file not found"
```bash
# Check if outputs/{config_name}/ exists:
ls -la outputs/M6/

# If missing, train first:
python main.py --data_path ... --config M6
```

### Issue: "Vocab mismatch"
- Check that char_vocab.json and best_model.pt are from same training run
- Ensure they're in the same output directory

### Issue: "Class mapping mismatch"
- Ensure class_mappings.json corresponds to the trained model
- Check idx_to_class has same tags as training data

### Issue: "OOM during inference"
```bash
# Reduce batch size:
python inference.py --config_name M6 --batch_size 8
```

### Issue: "Model architecture mismatch"
```bash
# Ensure char_type and use_crf match training config:
python inference.py --config_name M6 --char_type bilstm --use_crf true
```

---

## Performance Notes

- **Training**: ~20-30 min per scenario (on 1x GPU with 16GB VRAM)
- **Inference**: ~500-1000 sentences/sec (batch_size=32)
- **Output sizes**: 
  - Model checkpoint: ~500MB (BERT weights)
  - Vocab + mappings: <1MB
  - Pseudo-labeled CSV: ~50-100MB per 1M tokens

---

## Next Steps After Inference

1. **Correct predictions** in the pos_tag_koreksi column
2. **Combine with training data** to create augmented dataset
3. **Retrain model** with augmented data for improved performance
4. **Evaluate on held-out test set** to measure improvement
