# Modified Architecture - File Relationships & Process Flow

## Quick Start

### Training
```bash
python main.py --data_path ./data/sample-anotasi-merge-valid.csv --config M6
```
Outputs to: `outputs/M6/`

### Inference
```bash
python inference.py --config_name M6 --corpora_dir ./raw_corpora
```
Outputs to: `./data_pseudo_label.csv` (or split files)

---

## Core Files & Relationships

### main.py
**Purpose**: Training orchestrator that coordinates entire training pipeline

**Responsibilities**:
1. Load and prepare training data
2. Create train/val/test splits
3. Build and initialize HybridModel
4. Call train_model() for training
5. Evaluate on test set
6. Save all outputs:
   - best_model_{config}.pt
   - char_vocab_{config}.json
   - class_mappings_{config}.json
   - test_results_{config}.csv/json
   - classification_report_{config}.json
   - crf_transitions_{config}.csv (if use_crf=True)

**Calls**:
- `train_model()` from train.py
- `evaluate_with_tokens()` from train.py
- `save_test_results()` from train.py
- Model classes from feature_extraction.py

**Output Structure**:
```
outputs/{config_name}/
├── best_model_{config_name}.pt           ← For inference.py
├── char_vocab_{config_name}.json         ← For inference.py
├── class_mappings_{config_name}.json     ← For inference.py
├── test_results_{config_name}.csv
├── test_results_{config_name}.json
├── classification_report_{config_name}.json
└── crf_transitions_{config_name}.csv
```

---

### train.py
**Purpose**: Training utilities library with training loop and evaluation

**Key Functions**:

1. **train_model()**
   - Runs training loop with early stopping
   - Saves best checkpoint during training
   - Restores best weights before return
   - Called by main.py

2. **train_one_epoch()**
   - Single epoch training
   - Gradient computation and optimization step
   - Called by train_model()

3. **evaluate() / evaluate_with_tokens()**
   - Validation/test evaluation
   - Returns loss and predictions
   - evaluate_with_tokens() also returns token lists for confusion matrix
   - Called by train_model() and main.py

4. **save_test_results()**
   - Saves token-level predictions to CSV/JSON
   - Used by main.py after test evaluation
   - Outputs: token, true_label, pred_label, correctness

5. **compute_classification_report()**
   - Generates sklearn classification report
   - Saves to JSON format
   - Called by main.py

6. **build_optimizer()** / **build_scheduler()**
   - Creates AdamW optimizer with layer-wise learning rates
   - Creates linear warmup scheduler
   - Called by train_model()

**Relationships**:
- Used exclusively by main.py
- No direct dependency on inference.py
- Depends on feature_extraction.py for HybridModel

---

### inference.py
**Purpose**: Inference pipeline for pseudo-labeling unlabeled corpus

**Responsibilities**:
1. Load trained model outputs from main.py
2. Load unlabeled corpus
3. Run batch inference
4. Save predictions to CSV

**Key Functions**:

1. **load_model()**
   - Loads: best_model_{config}.pt, char_vocab_{config}.json, class_mappings_{config}.json
   - Returns: HybridModel, char_vocab, idx_to_class, tokenizer
   - Called by main()

2. **load_corpus_from_folder()**
   - Loads unlabeled text/WhatsApp corpus
   - Tokenizes and validates sentences
   - Returns list of (file_id, sentence_text) tuples

3. **run_inference()**
   - Main inference loop
   - Batches inference for efficiency
   - Saves outputs incrementally to CSV
   - Supports multiple output splitting strategies

4. **_predict_batch()**
   - Run model forward pass on batch
   - Returns predictions for each token
   - Handles BERT tokenization and char-level features

5. **_decode_preds()**
   - Converts prediction indices to POS tag names
   - Uses idx_to_class mapping from main.py

**Input Files** (from main.py outputs):
```
outputs/{config_name}/
├── best_model_{config_name}.pt           ← Model weights
├── char_vocab_{config_name}.json         ← Char vocabulary
└── class_mappings_{config_name}.json     ← Tag mappings
```

**Output Files**:
```
data_pseudo_label/
├── data_pseudo_label.csv (--split_by=none)
├── data_pseudo_label_{source}.csv (--split_by=file)
└── data_pseudo_label_part###.csv (--split_by=rows)

Columns: [file_id, sentence_id, token_index, token, pos_tag_pred, pos_tag_koreksi]
```

**New Features** (in this update):
- `--config_name` argument: auto-resolves paths from outputs/{config_name}/
- `--char_type` and `--use_crf` auto-loaded from config.yml
- Path validation with helpful error messages
- Comprehensive logging of configuration

---

### feature_extraction.py
**Purpose**: Model architecture components (BERT + Char-level + CRF)

**Key Classes**:

1. **HybridModel**
   - Main architecture: BERT → Char encoder → optional BiLSTM → optional CRF
   - Parameters controlled by:
     - char_type: "none", "cnn", "bilstm"
     - use_crf: bool
     - use_word_bilstm: bool

2. **CharCNN** / **CharBiLSTM**
   - Character-level feature extraction
   - Used when char_type != "none"
   - CharCNN: multi-kernel convolutions
   - CharBiLSTM: bidirectional LSTM on characters

3. **CRFLayer**
   - Conditional Random Field decoder
   - Learns transition scores between POS tags
   - Used when use_crf=True

**Used by**:
- main.py: Creates model instance for training
- inference.py: Loads pre-trained model for inference

---

### dataset.py
**Purpose**: PyTorch Dataset for POS tagging

**Key Classes**:
- **POSDataset**: Loads CSV data and prepares batches with char/BERT features
- **make_collate_fn()**: Batch collator with padding and masking

**Used by**:
- main.py: Creates train/val/test datasets

---

### preprocess.py
**Purpose**: Data preprocessing and vocabulary building

**Key Functions**:
- **build_char_vocab()**: Build character vocabulary
- **split_train_val_test()**: Data splitting
- **calculate_class_weights()**: Compute class weights for imbalanced data
- **check_vocab_coverage()**: Validate vocab on data

**Used by**:
- main.py: Preprocessing before training

---

### config.yml & config.py
**Purpose**: Configuration management for different training scenarios

**config.yml**: Defines scenarios M1-M6, M6a, M6b
- Each scenario specifies: model_name, char_type, use_crf, use_word_bilstm

**config.py**: Python dataclass configuration (alternative to YAML)

**Used by**:
- main.py: Loads scenario config when --config flag is used
- inference.py: Loads char_type/use_crf from config.yml when --config_name is used

---

## Process Flow Diagram

### Training Phase
```
data.csv
   ↓
main.py:
├─ load & split data
├─ build_char_vocab() → char_vocab.json
├─ create HybridModel
├─ train_model():                         [train.py]
│  ├─ train_one_epoch():                  [train.py]
│  │  ├─ forward pass
│  │  ├─ loss computation
│  │  └─ backprop + optimize
│  ├─ evaluate():                         [train.py]
│  │  ├─ validation loss
│  │  └─ validation accuracy
│  └─ early stopping logic
│
├─ evaluate_with_tokens():                [train.py]
│  ├─ test loss
│  ├─ predictions (preds)
│  ├─ ground truth (labels)
│  ├─ token list
│  └─ sentence indices
│
├─ save_test_results(preds, labels)       [train.py]
│  └─ test_results.csv/json
│
├─ save best_model.pt
├─ save char_vocab.json
├─ save class_mappings.json
├─ save classification_report.json
└─ save crf_transitions.csv (optional)
   
   ↓
outputs/{config_name}/ ✓
```

### Inference Phase
```
raw_corpora/
   ↓
inference.py:
├─ parse_args():
│  ├─ resolve config_name → output paths
│  └─ validate file existence
│
├─ load_model():                          [loads from outputs/{config_name}/]
│  ├─ best_model_{config}.pt
│  ├─ char_vocab_{config}.json
│  ├─ class_mappings_{config}.json
│  └─ BERT tokenizer
│
├─ load_corpus_from_folder()
│  └─ sentence list: [(file_id, text), ...]
│
├─ run_inference():
│  ├─ for each batch:
│  │  ├─ _predict_batch():
│  │  │  ├─ BERT tokenization
│  │  │  ├─ char-level features
│  │  │  ├─ forward pass → predictions
│  │  │  └─ return indices
│  │  └─ _decode_preds():
│  │     └─ indices → tag names
│  │
│  └─ _flush():
│     └─ write CSV incrementally
   
   ↓
data_pseudo_label.csv ✓
```

---

## Data & File Mapping

### Vocabulary & Mappings
```
char_vocab.json format:
{
  "a": 1,
  "b": 2,
  ...
  "unk": 0  // Unknown character
}
Size: Keys = all unique characters in training corpus
Usage: prepare_char_ids() in inference → char_ids tensor for char encoder
```

```
class_mappings.json format:
{
  "class_to_idx": {
    "NN": 0,
    "VB": 1,
    "JJ": 2,
    ...
  },
  "idx_to_class": {
    "0": "NN",
    "1": "VB",
    "2": "JJ",
    ...
  }
}
Size: num_classes = unique POS tags in training data
Usage: _decode_preds() in inference → convert prediction indices to tag names
```

### Test Results
```
test_results.csv:
sentence_id, token_idx, token, true_label, pred_label, correct, true_idx, pred_idx
1, 1, "Saya", "NN", "NN", True, 0, 0
1, 2, "makan", "VB", "NN", False, 1, 0
...

Used for: Confusion matrix analysis, error analysis, performance debugging
```

### Inference Output
```
data_pseudo_label.csv:
file_id, sentence_id, token_index, token, pos_tag_pred, pos_tag_koreksi
file1, 1, 1, "Saya", "NN", ""
file1, 1, 2, "makan", "VB", ""
...

pos_tag_koreksi: EMPTY → user fills with correction if pos_tag_pred is wrong
Used for: Creating augmented training data by correcting pseudo-labels
```

---

## Configuration Parameters & Their Impact

### Model Architecture (config.yml scenarios)
| Param | Values | Impact | Used in |
|-------|--------|--------|---------|
| char_type | "none" | No char features, lighter model | feature_extraction.py, HybridModel |
| | "cnn" | CNN-based char encoder (M4, M6a) | CharCNN class |
| | "bilstm" | BiLSTM char encoder (M5, M6) | CharBiLSTM class |
| use_crf | true/false | Enable CRF layer for joint decoding | CRFLayer class |
| use_word_bilstm | true/false | BiLSTM after subword pooling (M3) | WordBiLSTM class |
| model_name | HF identifier | BERT base model (M1-5: indobert, M6: indobert-tweet) | AutoModel.from_pretrained() |

### Inference Parameters
| Param | Default | Impact |
|-------|---------|--------|
| --config_name | "" | Auto-resolve paths from outputs/{config}/ |
| --batch_size | 32 | Tokens per inference batch (memory vs speed) |
| --split_by | "file" | Output splitting strategy |
| --char_type | "bilstm" | Must match training configuration |
| --use_crf | True | Must match training configuration |

---

## Error Prevention Checklist

- [ ] char_type in inference matches training config
- [ ] use_crf in inference matches training config  
- [ ] class_mappings.json has "idx_to_class" with string keys
- [ ] char_vocab.json maps characters to integers
- [ ] Output files exist before running inference
- [ ] BERT model_name is same in training and inference
- [ ] Training data has [token, pos_tag] columns
- [ ] Corpus files are in outputs/{config_name}/ for config_name usage

---

## Performance Optimization

### Training
- Use `--batch_size 32` for 16GB GPU
- Enable `--use_crf` for better accuracy (slightly slower)
- Use `--char_type bilstm` for morphology-rich languages
- Set `--patience 3` for early stopping

### Inference  
- Use `--batch_size 64` for faster inference (if VRAM allows)
- Use `--split_by rows` for very large corpora
- Disable BERT gradient computation (done automatically)
- Use `--char_type bilstm` for better accuracy (slightly slower)

---

## Debugging Tips

1. **Check output structure**: `ls -la outputs/M6/`
2. **Validate JSONs**: `python -m json.tool outputs/M6/char_vocab_m6.json`
3. **Check data shapes**: Add print statements in _predict_batch()
4. **Monitor inference**: Look for "Inference N/M kalimat" log messages
5. **Verify mappings**: Compare class_mappings.json with actual predictions

---

## References

- **WORKFLOW.md**: Complete end-to-end pipeline documentation
- **config.yml**: Scenario definitions (M1-M6, M6a, M6b)
- **ARCHITECTURE.md**: Detailed model architecture documentation (if exists)
