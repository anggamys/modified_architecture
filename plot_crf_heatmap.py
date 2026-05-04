import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
from typing import Optional

def plot_heatmap(csv_path: str, output_png: str = "heatmap_crf.png") -> None:
    print(f"Memuat data dari: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} tidak ditemukan!")
        print("Pastikan Anda sudah menjalankan ulang proses training/evaluasi di main.py untuk men-generate file ini.")
        return

    # 1. Load data CSV yang tadi disimpan
    df_loaded = pd.read_csv(csv_path, index_col=0)

    # 2. Filter hanya untuk tag penting (Bisa disesuaikan dengan kebutuhan)
    important_tags = ['PREP', 'N-KON', 'VB-T', 'VB-IT', 'ADJ', 'INTJ', 'ADV-TMP', 'N-NDR', 'PRT', 'CONJ-KRD']
    
    # Memastikan tag yang difilter memang ada di dalam dataframe
    valid_tags = [t for t in important_tags if t in df_loaded.columns and t in df_loaded.index]
    
    if len(valid_tags) == 0:
        print("Tag yang ditentukan tidak ditemukan, mem-plot 10 tag pertama sebagai contoh...")
        df_filtered = df_loaded.iloc[:10, :10]
    else:
        df_filtered = df_loaded.loc[valid_tags, valid_tags]

    # 3. Buat Heatmap
    plt.figure(figsize=(10, 8))
    # annot=True untuk menampilkan angka, cmap="Greys" untuk warna hitam putih
    sns.heatmap(df_filtered, annot=True, fmt=".2f", cmap="Greys", cbar=True)

    plt.title("CRF Transition Matrix (Internal Weights)", fontsize=14)
    plt.xlabel("To Tag", fontsize=12)
    plt.ylabel("From Tag", fontsize=12)

    # 4. Simpan langsung sebagai gambar PNG resolusi tinggi (Syarat jurnal/MIT Press)
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"\nHeatmap berhasil disimpan sebagai: {output_png}")
    
    # Tampilkan plot (Jika dijalankan di lokal/Jupyter)
    # Jika dijalankan di server tanpa GUI, Anda mungkin perlu me-comment baris ini
    try:
        plt.show()
    except Exception:
        print("Tidak dapat menampilkan UI gambar secara langsung. File PNG sudah tersimpan.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Heatmap untuk CRF Transition Matrix")
    parser.add_argument("--csv_path", type=str, default="training_result/crf_transitions.csv", help="Path ke file CSV matriks transisi")
    parser.add_argument("--output_png", type=str, default="analyst/heatmap_crf.png", help="Path output file PNG resolusi tinggi")
    args = parser.parse_args()
    
    plot_heatmap(args.csv_path, args.output_png)
