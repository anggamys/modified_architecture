import torch

from transformers import AutoTokenizer

from preprocess import prepare_char_ids
from feature_extraction import CharCNN, Bert, HybridModel
from utils import log, log_level, dowloadModel


def test_feature_extraction(
    sample_data,
    char_vocab,
    model_name,
    batch_size=32,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    log("Testing feature extraction", level=log_level.INFO)

    # Ambil sample batch
    sample_batch = sample_data.head(batch_size).copy()

    log(f"Sample batch size: {len(sample_batch)}", level=log_level.INFO)
    log(f"Device: {device}", level=log_level.INFO)

    char_extraction = CharCNN(vocab_size=len(char_vocab))
    char_extraction.to(device)
    char_extraction.eval()

    # Prepare char_ids: (B, S, W) dimana S adalah panjang sequence
    tokens = sample_batch["token"].astype(str).values
    char_ids = prepare_char_ids(tokens, char_vocab, max_word_len=50)

    # Reshape untuk batch: (1, B, W) -> (B, W) untuk single sequence
    char_ids_tensor = torch.from_numpy(char_ids).unsqueeze(0).to(device)  # (1, B, W)

    log(
        f"[CharCNN] Input shape (char_ids): {char_ids_tensor.shape} - (B, S, W)",
        level=log_level.INFO,
    )

    with torch.no_grad():
        char_output = char_extraction(char_ids_tensor)

    log(f"[CharCNN] Output shape: {char_output.shape}", level=log_level.INFO)
    log(f"[CharCNN] Output dtype: {char_output.dtype}", level=log_level.INFO)
    log(
        f"[CharCNN] Output mean: {char_output.mean().item():.6f}, std: {char_output.std().item():.6f}",
        level=log_level.INFO,
    )

    # Log sample output values
    log(
        f"[CharCNN] Sample output values (first 5): {char_output.view(-1)[:5].cpu().numpy()}",
        level=log_level.INFO,
    )

    model_path = dowloadModel(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    bert_extraction = Bert(model_name=model_path)
    bert_extraction.to(device)
    bert_extraction.eval()

    # Tokenize sample text (concat tokens as sentence)
    sample_text = " ".join(tokens[:10])  # Ambil 10 tokens pertama

    encoding = tokenizer(
        sample_text, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)

    log(f"[BERT] Input tokens: {len(tokens[:10])} tokens", level=log_level.INFO)
    log(
        f"[BERT] Tokenized length: {encoding['input_ids'].shape[1]}",
        level=log_level.INFO,
    )
    log(
        f"[BERT] Input shape (input_ids): {encoding['input_ids'].shape}",
        level=log_level.INFO,
    )
    log(
        f"[BERT] Attention mask shape: {encoding['attention_mask'].shape}",
        level=log_level.INFO,
    )

    with torch.no_grad():
        bert_output = bert_extraction(
            input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"]
        )

    log(f"[BERT] Output shape: {bert_output.shape}", level=log_level.INFO)
    log(f"[BERT] Output dtype: {bert_output.dtype}", level=log_level.INFO)
    log(
        f"[BERT] Output mean: {bert_output.mean().item():.6f}, std: {bert_output.std().item():.6f}",
        level=log_level.INFO,
    )

    # Log sample output values
    log(
        f"[BERT] Sample output values (first 5): {bert_output.view(-1)[:5].cpu().numpy()}",
        level=log_level.INFO,
    )

    num_classes = sample_data["pos_tag"].nunique()
    hybrid_model = HybridModel(
        char_vocab_size=len(char_vocab),
        num_classes=num_classes,
    ).to(device)
    hybrid_model.eval()

    log(f"Number of POS classes: {num_classes}", level=log_level.INFO)

    # Prepare labels untuk testing (dummy)
    # Labels shape: (B, S) - sesuai dengan attention_mask
    labels = torch.zeros(
        (encoding["input_ids"].shape[0], encoding["input_ids"].shape[1]),
        dtype=torch.long,
    ).to(device)

    log(f"[HybridModel] Labels shape: {labels.shape}", level=log_level.INFO)

    with torch.no_grad():
        # Test forward pass dengan labels (untuk loss calculation)
        loss = hybrid_model(
            char_ids=char_ids_tensor,
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            labels=labels,
        )

    log(f"[HybridModel] Loss (with labels): {loss.item():.6f}", level=log_level.INFO)

    # Test inference mode (tanpa labels)
    with torch.no_grad():
        preds = hybrid_model(
            char_ids=char_ids_tensor,
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            labels=None,
        )

    log(f"[HybridModel] Predictions: {preds}", level=log_level.INFO)
    log(
        f"[HybridModel] Number of predicted sequences: {len(preds)}",
        level=log_level.INFO,
    )

    if len(preds) > 0:
        log(
            f"[HybridModel] First sequence length: {len(preds[0])}",
            level=log_level.INFO,
        )

        log(
            f"[HybridModel] First sequence predictions: {preds[0][:10]}",
            level=log_level.INFO,
        )

    # ===== Summary =====
    log("Testing Summary", level=log_level.INFO)
    log(f"✓ CharCNN output shape: {char_output.shape}", level=log_level.INFO)
    log(f"✓ BERT output shape: {bert_output.shape}", level=log_level.INFO)
    log(f"✓ HybridModel loss: {loss.item():.6f}", level=log_level.INFO)
    log(f"✓ HybridModel predictions: {len(preds)} sequences", level=log_level.INFO)
    log("✓ All models tested successfully!", level=log_level.INFO)

    return char_extraction, bert_extraction, hybrid_model
