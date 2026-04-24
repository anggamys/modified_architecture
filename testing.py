import torch
import numpy as np
import pandas as pd

from torch import Tensor
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)

from preprocess import prepare_char_ids
from utils import log, log_level, dowloadModel
from feature_extraction import CharCNN, Bert, HybridModel


def test_feature_extraction(
    sample_data: pd.DataFrame,
    char_vocab: dict[str, int],
    model_name: str,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[CharCNN, Bert, HybridModel]:
    log(domain="Testing", msg="Testing feature extraction", level=log_level.INFO)

    # Ambil sample batch
    sample_batch: pd.DataFrame = sample_data.head(batch_size).copy()

    log(
        domain="Testing",
        msg=f"Sample batch size: {len(sample_batch)}",
        level=log_level.INFO,
    )
    log(
        domain="Testing",
        msg=f"Device: {device}",
        level=log_level.INFO,
    )

    char_extraction: CharCNN = CharCNN(vocab_size=len(char_vocab))
    char_extraction.to(device)
    char_extraction.eval()

    # Prepare char_ids: (B, S, W) dimana S adalah panjang sequence
    tokens: np.ndarray = sample_batch["token"].astype(str).values
    char_ids: np.ndarray = prepare_char_ids(tokens, char_vocab, max_word_len=50)

    # Reshape untuk batch: (1, B, W) -> (B, W) untuk single sequence
    char_ids_tensor: Tensor = (
        torch.from_numpy(char_ids).unsqueeze(0).to(device)
    )  # (1, B, W)

    log(
        domain="Testing",
        msg=f"[CharCNN] Input shape (char_ids): {char_ids_tensor.shape} - (B, S, W)",
        level=log_level.INFO,
    )

    with torch.no_grad():
        char_output: Tensor = char_extraction(char_ids_tensor)

    log(
        domain="Testing",
        msg=f"[CharCNN] Output shape: {char_output.shape}",
        level=log_level.INFO,
    )
    log(
        domain="Testing",
        msg=f"[CharCNN] Output dtype: {char_output.dtype}",
        level=log_level.INFO,
    )
    log(
        domain="Testing",
        msg=f"[CharCNN] Output mean: {char_output.mean().item():.6f}, std: {char_output.std().item():.6f}",
        level=log_level.INFO,
    )

    # Log sample output values
    log(
        domain="Testing",
        msg=f"[CharCNN] Sample output values (first 5): {char_output.view(-1)[:5].cpu().numpy()}",
        level=log_level.INFO,
    )

    model_path: str = dowloadModel(model_name)
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_path)

    # dowloadModel mengembalikan path string — perlu dimuat dulu dengan AutoModel
    bert_model: PreTrainedModel = AutoModel.from_pretrained(model_path)
    bert_extraction: Bert = Bert(bert=bert_model)
    bert_extraction.to(device)
    bert_extraction.eval()

    # Tokenize token list (word-level) dengan is_split_into_words=True
    # agar tokenizer bisa mengembalikan word_ids() — mapping subword → word index
    tokens_list: list[str] = tokens[:10].tolist()  # Ambil 10 tokens pertama

    encoding = tokenizer(
        tokens_list,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    # word_ids harus diambil sebelum .to(device) — ini Python list, bukan tensor
    word_ids: list[list[int | None]] = [encoding.word_ids(batch_index=0)]

    encoding = encoding.to(device)
    input_ids: Tensor = encoding["input_ids"]
    attention_mask: Tensor = encoding["attention_mask"]

    log(
        domain="Testing",
        msg=f"[BERT] Input tokens: {len(tokens[:10])} tokens",
        level=log_level.INFO,
    )
    log(
        domain="Testing",
        msg=f"[BERT] Tokenized length: {input_ids.shape[1]}",
        level=log_level.INFO,
    )
    log(
        domain="Testing",
        msg=f"[BERT] Input shape (input_ids): {input_ids.shape}",
        level=log_level.INFO,
    )
    log(
        domain="Testing",
        msg=f"[BERT] Attention mask shape: {attention_mask.shape}",
        level=log_level.INFO,
    )

    with torch.no_grad():
        bert_output: Tensor = bert_extraction(
            input_ids=input_ids, attention_mask=attention_mask
        )

    log(
        domain="Testing",
        msg=f"[BERT] Output shape: {bert_output.shape}",
        level=log_level.INFO,
    )
    log(
        domain="Testing",
        msg=f"[BERT] Output dtype: {bert_output.dtype}",
        level=log_level.INFO,
    )
    log(
        domain="Testing",
        msg=f"[BERT] Output mean: {bert_output.mean().item():.6f}, std: {bert_output.std().item():.6f}",
        level=log_level.INFO,
    )

    # Log sample output values
    log(
        domain="Testing",
        msg=f"[BERT] Sample output values (first 5): {bert_output.view(-1)[:5].cpu().numpy()}",
        level=log_level.INFO,
    )

    num_classes: int = sample_data["pos_tag"].nunique()
    hybrid_model: HybridModel = HybridModel(
        char_vocab_size=len(char_vocab),
        num_classes=num_classes,
        bert=bert_model,
    ).to(device)
    hybrid_model.eval()

    log(
        domain="Testing",
        msg=f"Number of POS classes: {num_classes}",
        level=log_level.INFO,
    )

    # Labels harus di level kata (word-level), bukan subword.
    # _align_labels_to_bert akan memetakannya ke subword menggunakan word_ids.
    labels: Tensor = torch.zeros(
        (1, len(tokens_list)),
        dtype=torch.long,
    ).to(device)

    log(
        domain="Testing",
        msg=f"[HybridModel] Labels shape: {labels.shape}",
        level=log_level.INFO,
    )

    with torch.no_grad():
        # Test forward pass dengan labels (untuk loss calculation)
        loss: Tensor = hybrid_model(
            char_ids=char_ids_tensor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_ids=word_ids,
            labels=labels,
        )

    log(
        domain="Testing",
        msg=f"[HybridModel] Loss (with labels): {loss.item():.6f}",
        level=log_level.INFO,
    )

    # Test inference mode (tanpa labels)
    with torch.no_grad():
        preds: Tensor = hybrid_model(
            char_ids=char_ids_tensor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_ids=word_ids,
            labels=None,
        )

    log(
        domain="Testing",
        msg=f"[HybridModel] Predictions: {preds}",
        level=log_level.INFO,
    )
    log(
        domain="Testing",
        msg=f"[HybridModel] Number of predicted sequences: {len(preds)}",
        level=log_level.INFO,
    )

    if len(preds) > 0:
        log(
            domain="Testing",
            msg=f"[HybridModel] First sequence length: {len(preds[0])}",
            level=log_level.INFO,
        )
        log(
            domain="Testing",
            msg=f"[HybridModel] First sequence predictions: {preds[0][:10]}",
            level=log_level.INFO,
        )

    log(
        domain="Testing",
        msg="Testing Summary",
        level=log_level.INFO,
    )
    log(
        domain="Testing",
        msg=f"CharCNN output shape: {char_output.shape}",
        level=log_level.INFO,
    )
    log(
        domain="Testing",
        msg=f"BERT output shape: {bert_output.shape}",
        level=log_level.INFO,
    )
    log(
        domain="Testing",
        msg=f"HybridModel loss: {loss.item():.6f}",
        level=log_level.INFO,
    )
    log(
        domain="Testing",
        msg=f"HybridModel predictions: {len(preds)} sequences",
        level=log_level.INFO,
    )
    log(
        domain="Testing",
        msg="All models tested successfully!",
        level=log_level.INFO,
    )

    return char_extraction, bert_extraction, hybrid_model
