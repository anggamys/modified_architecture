import json
import re

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
    # Sesuaikan dengan path file json Anda
    filepath = 'training_result/test_results.json'
    explore_data(filepath)
