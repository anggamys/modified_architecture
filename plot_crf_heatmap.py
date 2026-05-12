import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import glob

# Research-ready Premium Grayscale style
plt.rcParams.update({
    "figure.facecolor":  "#ffffff",
    "axes.facecolor":    "#ffffff",
    "axes.edgecolor":    "#333333",
    "axes.labelcolor":   "#000000",
    "axes.titlesize":    16,
    "axes.titleweight":  "bold",
    "axes.labelsize":    12,
    "xtick.color":       "#333333",
    "ytick.color":       "#333333",
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "grid.color":        "#dddddd",
    "grid.linewidth":    0.8,
    "grid.linestyle":    "--",
    "legend.fontsize":   10,
    "legend.frameon":    True,
    "legend.edgecolor":  "#333333",
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "DejaVu Sans", "Helvetica"],
    "savefig.dpi":       300, 
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

def resolve_m6_csv(base_dir):
    """Try to find the CRF transitions CSV for model M6."""
    # 1. Try direct path
    direct_path = os.path.join(base_dir, "crf_transitions_m6.csv")
    if os.path.exists(direct_path):
        return direct_path
    
    # 2. Try drive subfolder
    drive_pattern = os.path.join(base_dir, "drive*-m6", "crf_transitions_m6.csv")
    matches = glob.glob(drive_pattern)
    if matches:
        return matches[0]
        
    # 3. Fallback to default name if M6 specific not found
    fallback_path = os.path.join(base_dir, "crf_transitions.csv")
    return fallback_path

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
    plt.figure(figsize=(12, 10))
    
    # anot=True untuk menampilkan angka, cmap="Greys" untuk warna hitam putih
    # Tambahkan garis antar sel untuk estetika premium
    sns.heatmap(df_filtered, annot=True, fmt=".2f", cmap="Greys", 
                cbar=True, linewidths=.5, linecolor='lightgray',
                annot_kws={"size": 9, "weight": "bold"})

    plt.title("CRF Transition Matrix - Model M6 (Internal Weights)", pad=20)
    plt.xlabel("To Tag", labelpad=15)
    plt.ylabel("From Tag", labelpad=15)

    # Simpan langsung sebagai gambar PNG resolusi tinggi
    plt.savefig(output_png)
    print(f"\nHeatmap berhasil disimpan sebagai: {output_png}")
    
    # Tampilkan plot (Jika dijalankan di lokal/Jupyter)
    # Jika dijalankan di server tanpa GUI, Anda mungkin perlu me-comment baris ini
    try:
        plt.show()
    except Exception:
        print("Tidak dapat menampilkan UI gambar secara langsung. File PNG sudah tersimpan.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Heatmap untuk CRF Transition Matrix")
    parser.add_argument("--base_dir", type=str, default="training_result", help="Folder utama training result")
    parser.add_argument("--csv_path", type=str, help="Path ke file CSV (opsional, akan dicari otomatis jika kosong)")
    parser.add_argument("--output_png", type=str, default="analyst/heatmap_crf_m6.png", help="Path output file PNG")
    args = parser.parse_args()
    
    csv_path = args.csv_path or resolve_m6_csv(args.base_dir)
    
    plot_heatmap(csv_path, args.output_png)
