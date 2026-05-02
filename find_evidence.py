import json
import argparse
import os

def load_json(path):
    if not os.path.exists(path):
        print(f"Peringatan: File {path} tidak ditemukan.")
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_tokens_from_report(report):
    if not report:
        return {}
    tokens_dict = {}
    for sent in report.get("predictions", []):
        sent_id = sent.get("sentence_id")
        for token_data in sent.get("tokens", []):
            token_idx = token_data.get("token_idx")
            tokens_dict[(sent_id, token_idx)] = token_data
    return tokens_dict

def find_evidence(m4_path, m6_path, m1_path=None):
    m4_data = load_json(m4_path)
    m6_data = load_json(m6_path)
    
    if not m4_data or not m6_data:
        print("Tidak bisa melanjutkan karena data hasil prediksi tidak lengkap.")
        return
    
    m4_tokens = extract_tokens_from_report(m4_data)
    
    m1_tokens = {}
    if m1_path:
        m1_data = load_json(m1_path)
        m1_tokens = extract_tokens_from_report(m1_data)
        
    n_abs_cases = []
    vb_t_cases = []
    domain_gap_cases = []
    
    for sent in m6_data.get("predictions", []):
        sent_id = sent.get("sentence_id")
        tokens = sent.get("tokens", [])
        
        full_sent = " ".join([t["token"] for t in tokens])
        
        for t in tokens:
            token_idx = t["token_idx"]
            token_text = t["token"].lower()
            true_label = t["true_label"]
            
            key = (sent_id, token_idx)
            if key not in m4_tokens:
                continue
                
            m4_pred = m4_tokens[key]["pred_label"]
            m6_pred = t["pred_label"]
            
            m4_correct = m4_tokens[key]["correct"]
            m6_correct = t["correct"]
            
            m1_pred = m1_tokens[key]["pred_label"] if key in m1_tokens else "N/A"
            m1_correct = m1_tokens[key]["correct"] if key in m1_tokens else False
            
            # Case 1: N-ABS
            # Kata-kata panjang yang berakhiran -an (terutama awalan pe- atau ke-) 
            # Dimana M4 salah tebak tapi M6 benar.
            if true_label == "N-ABS" and not m4_correct and m6_correct:
                if len(token_text) > 5 and (token_text.endswith("an") or token_text.endswith("nya")):
                   n_abs_cases.append({
                       "token": t["token"],
                       "sentence": full_sent,
                       "m1_pred": m1_pred,
                       "m4_pred": m4_pred,
                       "m6_pred": m6_pred,
                       "true_label": true_label
                   })
                   
            # Case 2: VB-T (atau VB-IT) dengan awalan ng-/ny-/nge-
            # Dimana M4 salah tebak tapi M6 benar.
            if (true_label == "VB-T" or true_label == "VB-IT") and not m4_correct and m6_correct:
                if token_text.startswith("ng") or token_text.startswith("ny") or token_text.startswith("nge"):
                   vb_t_cases.append({
                       "token": t["token"],
                       "sentence": full_sent,
                       "m1_pred": m1_pred,
                       "m4_pred": m4_pred,
                       "m6_pred": m6_pred,
                       "true_label": true_label
                   })
                   
            # Case 3: Domain Gap (Kosakata Slang/Informal)
            # Dimana M1 (IndoBERT-base) gagal, tapi M6 (IndoBERTweet) berhasil.
            if not m1_correct and m6_correct:
                # Filter kata-kata yang umum menjadi slang atau partikel
                token_lower = token_text.lower()
                slang_keywords = ["wkwk", "haha", "anjir", "kuy", "gemoy", "rl", "rp", "mutualan", "bgt", "sih", "dong", "deh", "loh", "kok", "kek", "kalo", "gak", "yg", "dgn", "utk", "jd", "udh", "aja"]
                is_slang = any(keyword in token_lower for keyword in slang_keywords) or true_label in ["INTJ", "PRT", "ADV-CND", "ADV-DEG", "ADV-TMP", "ADV-LOC"]
                
                if is_slang:
                    domain_gap_cases.append({
                        "token": t["token"],
                        "sentence": full_sent,
                        "m1_pred": m1_pred,
                        "m4_pred": m4_pred,
                        "m6_pred": m6_pred,
                        "true_label": true_label
                    })

    print("-"*60)
    print(" BUKTI KASUS 1: N-ABS (Konfiks panjang misal ke-...-an / pe-...-an) ")
    print("-"*60)
    print(f"Ditemukan {len(n_abs_cases)} kasus dimana M4 (CNN) gagal tapi M6 (BiLSTM) berhasil.")
    for i, case in enumerate(n_abs_cases[:10]): 
        print(f"\n{i+1}. Token    : {case['token']}")
        print(f"   Kalimat  : {case['sentence']}")
        print(f"   True     : {case['true_label']}")
        print(f"   M1 (Base): {case['m1_pred']}")
        print(f"   M4 (CNN) : {case['m4_pred']} (SALAH)")
        print(f"   M6 (LSTM): {case['m6_pred']} (BENAR)")

    print("\n" + "-"*60)
    print(" BUKTI KASUS 2: VB-T / VB-IT (Nasalisasi ng- / ny-) ")
    print("-"*60)
    print(f"Ditemukan {len(vb_t_cases)} kasus dimana M4 (CNN) gagal tapi M6 (BiLSTM) berhasil.")
    for i, case in enumerate(vb_t_cases[:10]): 
        print(f"\n{i+1}. Token    : {case['token']}")
        print(f"   Kalimat  : {case['sentence']}")
        print(f"   True     : {case['true_label']}")
        print(f"   M1 (Base): {case['m1_pred']}")
        print(f"   M4 (CNN) : {case['m4_pred']} (SALAH)")
        print(f"   M6 (LSTM): {case['m6_pred']} (BENAR)")
        
    print("\n" + "-"*60)
    print(" BUKTI KASUS 3: Kesenjangan Domain (Slang/Informal) ")
    print("-"*60)
    print(f"Ditemukan {len(domain_gap_cases)} kasus slang dimana M1 (IndoBERT-base) gagal tapi M6 (IndoBERTweet) berhasil.")
    for i, case in enumerate(domain_gap_cases[:15]): 
        print(f"\n{i+1}. Token    : {case['token']}")
        print(f"   Kalimat  : {case['sentence']}")
        print(f"   True     : {case['true_label']}")
        print(f"   M1 (Base): {case['m1_pred']} (SALAH)")
        print(f"   M6 (LSTM): {case['m6_pred']} (BENAR)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find qualitative evidence comparing M4 and M6 predictions.")
    parser.add_argument("--m4_path", default="training_result/test_results_m4.json", help="Path to M4 test results JSON")
    parser.add_argument("--m6_path", default="training_result/test_results_m6.json", help="Path to M6 test results JSON")
    parser.add_argument("--m1_path", default="training_result/test_results_m1.json", help="Path to M1 test results JSON")
    args = parser.parse_args()
    
    find_evidence(args.m4_path, args.m6_path, args.m1_path)
