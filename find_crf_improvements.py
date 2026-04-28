import json
import os

def load_data_flat(filepath):
    if not os.path.exists(filepath):
        return []
        
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    predictions_raw = data.get('predictions', [])
    
    flat_tokens = []
    if predictions_raw and isinstance(predictions_raw[0], dict):
        if "tokens" in predictions_raw[0]:
            for sent in predictions_raw:
                flat_tokens.extend(sent["tokens"])
        else:
            flat_tokens = predictions_raw
    elif predictions_raw and isinstance(predictions_raw[0], list):
        for sent in predictions_raw:
            flat_tokens.extend(sent)
            
    return flat_tokens

def compare_models(file_m1, file_m6):
    if not os.path.exists(file_m1) or not os.path.exists(file_m6):
        print(f"Error: Pastikan file {file_m1} dan {file_m6} tersedia.")
        return

    # Load semua prediksi M1 ke dalam flat array (agar kompatibel jika M1 masih format lama)
    m1_flat = load_data_flat(file_m1)
    
    # Load M6 untuk mendapatkan pembatas kalimat (sentence boundaries)
    with open(file_m6, 'r', encoding='utf-8') as f:
        data_m6 = json.load(f)
    
    m6_predictions = data_m6.get('predictions', [])
    
    results_to_show = []
    flat_idx = 0
    
    for sent_idx, sent_data in enumerate(m6_predictions):
        # Cek format data M6
        if isinstance(sent_data, dict) and "tokens" in sent_data:
            m6_tokens = sent_data["tokens"]
            s_id = sent_data.get("sentence_id", sent_idx)
        elif isinstance(sent_data, list):
            m6_tokens = sent_data
            s_id = sent_idx
        else:
            print("Error: test_results_m6.json tidak memiliki pembagian kalimat. Silakan jalankan ulang main.py untuk M6.")
            return
            
        sent_len = len(m6_tokens)
        m1_tokens = m1_flat[flat_idx : flat_idx + sent_len]
        flat_idx += sent_len
        
        # Hitung error pada masing-masing model di kalimat ini
        m1_errors = sum(1 for t in m1_tokens if t["true_label"] != t["pred_label"])
        m6_errors = sum(1 for t in m6_tokens if t["true_label"] != t["pred_label"])
        
        # Kita mencari kalimat di mana M1 melakukan kesalahan, tapi M6 berhasil memperbaiki sebagian besarnya
        if m1_errors > 0 and m6_errors < m1_errors:
            table_data = []
            for t1, t6 in zip(m1_tokens, m6_tokens):
                table_data.append({
                    "Token": t1["token"],
                    "True Label": t1["true_label"],
                    "Prediksi M1": t1["pred_label"],
                    "Prediksi M6": t6["pred_label"]
                })
            
            results_to_show.append({
                "sentence_id": s_id,
                "m1_errors": m1_errors,
                "m6_errors": m6_errors,
                "improvement": m1_errors - m6_errors,
                "table": table_data
            })
            
    # Urutkan berdasarkan kalimat yang paling banyak diperbaiki oleh CRF
    results_to_show.sort(key=lambda x: x["improvement"], reverse=True)
    
    print("=== ANALISIS PERBANDINGAN M1 (Linear) vs M6 (CRF) ===\n")
    print(f"Ditemukan {len(results_to_show)} kalimat di mana M6 (CRF) memperbaiki kesalahan sintaksis M1.\n")
    
    # Tampilkan 3 kalimat terbaik untuk dimasukkan ke jurnal
    for idx, res in enumerate(results_to_show[:3]):
        print(f"📌 Kandidat Jurnal {idx+1} (Sentence ID: {res['sentence_id']})")
        print(f"   Kesalahan M1: {res['m1_errors']} token -> M6: {res['m6_errors']} token (CRF memperbaiki {res['improvement']} kesalahan)")
        print("-" * 75)
        print(f"   | {'Token':<20} | {'True Label':<12} | {'Prediksi M1':<12} | {'Prediksi M6':<12} |")
        print("-" * 75)
        for row in res['table']:
            # Highlight jika M1 salah tapi M6 benar
            is_fixed = (row["True Label"] != row["Prediksi M1"]) and (row["True Label"] == row["Prediksi M6"])
            marker = "⭐" if is_fixed else "  "
            print(f"{marker} | {row['Token']:<20} | {row['True Label']:<12} | {row['Prediksi M1']:<12} | {row['Prediksi M6']:<12} |")
        print("-" * 75)
        print("Keterangan: ⭐ = M1 mengacaukan tata bahasa (salah prediksi), tapi CRF (M6) berhasil merapikannya sesuai aturan sintaksis.\n")
        
if __name__ == "__main__":
    file_m1 = "training_result/test_results_m1.json"
    file_m6 = "training_result/test_results_m6.json"
    compare_models(file_m1, file_m6)
