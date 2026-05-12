import json
import re
import os
import glob
import argparse

def resolve_model_path(base_dir, model_name, filename_template):
    """Resolve path to model data, checking base dir and drive subfolders."""
    # 1. Try direct path
    direct_path = os.path.join(base_dir, filename_template.format(model_name.lower()))
    if os.path.exists(direct_path):
        return direct_path
        
    # 2. Try drive subfolders
    drive_pattern = os.path.join(base_dir, f"drive*-{model_name.lower()}", filename_template.format(model_name.lower()))
    matches = glob.glob(drive_pattern)
    if matches:
        return matches[0]
        
    return direct_path # Fallback to original expected path

def explore_data(filepath):
    print(f"Membaca data dari: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {filepath} tidak ditemukan.")
        return
    
    predictions_raw = data.get('predictions', [])
    
    # Menangani format flat maupun format grouped per sentence
    predictions = []
    if predictions_raw and isinstance(predictions_raw[0], dict) and "tokens" in predictions_raw[0]:
        for sent in predictions_raw:
            predictions.extend(sent["tokens"])
    elif predictions_raw and isinstance(predictions_raw[0], list):
        for sent in predictions_raw:
            predictions.extend(sent)
    else:
        predictions = predictions_raw
    
    # 1. True_Label = INTJ, Predicted_Label is WRONG
    wrong_intj = [p for p in predictions if p.get('true_label') == 'INTJ' and not p.get('correct')]
    
    # 2. True_Label = UNID, Predicted_Label is WRONG
    wrong_unid = [p for p in predictions if p.get('true_label') == 'UNID' and not p.get('correct')]
    
    # 3. Pengulangan huruf ekstrem (elongasi) yang salah ditebak
    # Mencari huruf yang berulang 3 kali atau lebih (misal: "Maokkkkkk", "ayooo")
    elongation_pattern = re.compile(r'([a-zA-Z])\1{2,}', re.IGNORECASE)
    wrong_elongation = [p for p in predictions if elongation_pattern.search(p.get('token', '')) and not p.get('correct')]
    
    # Menampilkan hasil eksplorasi
    
    print(f"\n{'-'*60}")
    print(f"1. Data True Label = INTJ tapi Salah Prediksi ({len(wrong_intj)} tokens)")
    print(f"{'-'*60}")
    if not wrong_intj:
        print("Tidak ada data yang memenuhi kriteria.")
    for p in wrong_intj:
        print(f"Token: {p['token']:<20} | True: {p['true_label']:<8} | Pred: {p['pred_label']}")
        
    print(f"\n{'-'*60}")
    print(f"2. Data True Label = UNID tapi Salah Prediksi ({len(wrong_unid)} tokens)")
    print(f"{'-'*60}")
    if not wrong_unid:
        print("Tidak ada data yang memenuhi kriteria.")
    for p in wrong_unid:
        print(f"Token: {p['token']:<20} | True: {p['true_label']:<8} | Pred: {p['pred_label']}")
        
    print(f"\n{'-'*60}")
    print(f"3. Data dengan Elongasi Ekstrem tapi Salah Prediksi ({len(wrong_elongation)} tokens)")
    print(f"{'-'*60}")
    if not wrong_elongation:
        print("Tidak ada data yang memenuhi kriteria.")
    for p in wrong_elongation:
        print(f"Token: {p['token']:<20} | True: {p['true_label']:<8} | Pred: {p['pred_label']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore prediction errors in model results.")
    parser.add_argument("--base_dir", default="training_result", help="Base directory for results")
    parser.add_argument("--model", default="m6", help="Model name (e.g., m1, m4, m6)")
    parser.add_argument("--filepath", help="Direct path to json file (optional)")
    
    args = parser.parse_args()
    
    # Resolve path if not explicitly provided
    filepath = args.filepath or resolve_model_path(args.base_dir, args.model, "test_results_{}.json")
    
    explore_data(filepath)
