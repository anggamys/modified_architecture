#!/usr/bin/env python3
"""
POS TAGGING ANNOTATION DATA EXPLORATION (EDA)
--------------------------------------------
This script performs a comprehensive analysis of the POS tagging annotation dataset.
It generates statistical summaries and high-quality grayscale visualizations 
suitable for research publication.

Features:
- Basic statistical summary (tokens, sentences, vocabulary).
- Distribution of all POS tags and main linguistic categories.
- Sentence and token length analysis.
- Text noise profiling (OOV rate, elongation, and informal nasalization).
- Visualizations saved at 300 DPI with research-ready aesthetics.
"""

import warnings
import os
import textwrap
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────
# Global Configuration & Styles
# ────────────────────────────────────────────────────────────
CSV_PATH = "data/sample-anotasi-merge-valid-with-kbbi.csv"
OUT_DIR  = "eda_output"
os.makedirs(OUT_DIR, exist_ok=True)

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

# Master mapping for linguistic categories
POS_CATEGORY_MAP = {
    'VB': 'Verba',
    'ADJ': 'Adjektiva',
    'ADV': 'Adverbia',
    'N': 'Nomina',
    'PR': 'Pronomina',
    'NUM': 'Numeralia',
    'PREP': 'Preposisi',
    'CONJ': 'Konjungsi',
    'INTJ': 'Interjeksi',
    'ART': 'Artikula',
    'PRT': 'Partikel',
    'ONOMA': 'Onomatope',
    'BND': 'Bentuk Terikat',
    'UNID': 'Unidentified'
}

PUNC_TAGS = ['.', ',', ':', '``', "''", '-LRB-', '-RRB-', '"', '“', '”']

# ────────────────────────────────────────────────────────────
# 1. Load & Preprocessing
# ────────────────────────────────────────────────────────────
print("=" * 60)
print("  POS TAGGING ANNOTATION DATA EXPLORATION")
print("=" * 60)

if not os.path.exists(CSV_PATH):
    print(f"  [ERROR] CSV file not found at: {CSV_PATH}")
    exit(1)

df = pd.read_csv(CSV_PATH)

# Clean whitespace
df["token"]   = df["token"].astype(str).str.strip()
df["pos_tag"] = df["pos_tag"].astype(str).str.strip()

# Derived columns
df["source_file"] = df["global_sentence_id"].str.extract(r"(file_\w+?_txt\d+)_")
df["region"]      = df["global_sentence_id"].str.extract(r"file_(\w+?)_txt")

def get_main_tag(tag):
    tag = str(tag).strip()
    if tag.startswith('-') or tag in PUNC_TAGS:
        return 'Punctuation'
    main_code = tag.split('-')[0]
    return POS_CATEGORY_MAP.get(main_code, main_code)

df["pos_main"]  = df["pos_tag"].apply(get_main_tag)
df["token_len"] = df["token"].str.len()

# Validation: Check for unexpected tags
all_valid_prefixes = list(POS_CATEGORY_MAP.keys())
def is_valid_tag(tag):
    if tag in PUNC_TAGS or tag.startswith('-'):
        return True
    return any(tag.startswith(p) for p in all_valid_prefixes)

df["is_valid"] = df["pos_tag"].apply(is_valid_tag)
invalid_tags   = df[~df["is_valid"]]["pos_tag"].unique()

# Basic counts
n_tokens    = len(df)
n_sentences = df["global_sentence_id"].nunique()
n_pos_tags  = df["pos_tag"].nunique()
n_files     = df["source_file"].nunique()
n_regions   = df["region"].nunique()
n_vocab     = df["token"].nunique()

# ────────────────────────────────────────────────────────────
# 2. Basic statistical summary – PRINT TO CONSOLE
# ────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("  A. CORPUS SUMMARY")
print(f"{'─'*60}")
print(f"  Total tokens              : {n_tokens:,}")
print(f"  Total unique sentences    : {n_sentences:,}")
print(f"  Unique vocabulary (types) : {n_vocab:,}")
print(f"  Avg sentence length       : {n_tokens/n_sentences:.2f} tokens/sentence")
print(f"  Total source files        : {n_files:,}")
print(f"  Total regions             : {n_regions:,}")
print(f"  Unique POS label types    : {n_pos_tags:,}")

if len(invalid_tags) > 0:
    print(f"  [WARNING] Found {len(invalid_tags)} tags not in the valid list: {invalid_tags}")
else:
    print("  [SUCCESS] All POS tags are consistent with the valid codes list.")

tokens_per_sent = df.groupby("global_sentence_id").size()
print("\n  Sentence length statistics (tokens per sentence):")
print(textwrap.indent(str(tokens_per_sent.describe().round(2)), "    "))

print("\n  Token length statistics (number of characters):")
print(textwrap.indent(str(df["token_len"].describe().round(2)), "    "))

# ────────────────────────────────────────────────────────────
# B. POS Tag Distribution – Print to Console
# ────────────────────────────────────────────────────────────
pos_freq = df["pos_tag"].value_counts()

print(f"\n{'─'*60}")
print("  B. POS TAG DISTRIBUTION")
print(f"{'─'*60}")
print("\n  Top-10 Most Frequent POS Labels:")
top10_pos = pos_freq.head(10)
for label, cnt in top10_pos.items():
    bar = "█" * int(cnt / pos_freq.max() * 30)
    pct = cnt / n_tokens * 100
    print(f"    {label:<18} {cnt:>6,}  ({pct:5.2f}%)  {bar}")

print("\n  Bottom-5 Least Frequent POS Labels:")
bottom5_pos = pos_freq.tail(5)
for label, cnt in bottom5_pos.items():
    bar = "░" * max(1, int(cnt / pos_freq.max() * 30))
    pct = cnt / n_tokens * 100
    print(f"    {label:<18} {cnt:>6,}  ({pct:5.2f}%)  {bar}")

# ────────────────────────────────────────────────────────────
# 3. Summary table per source file
# ────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("  SUMMARY PER SOURCE FILE")
print(f"{'─'*60}")
summary_file = (
    df.groupby("source_file")
    .agg(
        total_token   = ("token", "count"),
        total_kalimat = ("global_sentence_id", "nunique"),
        jenis_pos     = ("pos_tag", "nunique"),
    )
    .reset_index()
)
summary_file["rerata_token_per_kalimat"] = (summary_file["total_token"] / summary_file["total_kalimat"]).round(2)
summary_file.columns = ["Source File", "Total Tokens", "Total Sentences", "POS Types", "Avg Tokens/Sentence"]
print(summary_file.to_string(index=False))
summary_file.to_csv(f"{OUT_DIR}/tabel_ringkasan_file.csv", index=False)
print(f"\n  [Saved] {OUT_DIR}/tabel_ringkasan_file.csv")

# ────────────────────────────────────────────────────────────
# 4. CHART 1 – Overall POS Tag Distribution
# ────────────────────────────────────────────────────────────
pos_counts = df["pos_tag"].value_counts()
fig, ax = plt.subplots(figsize=(10, max(6, len(pos_counts) * 0.28)))
# Alternating colors and hatches for "premium" look
colors = ["#444444", "#777777", "#aaaaaa"]
hatches = ["", "///", "\\\\"]
bars = ax.barh(pos_counts.index[::-1], pos_counts.values[::-1], color="#ffffff", edgecolor="black", linewidth=0.8)

for i, bar in enumerate(bars):
    idx = i % len(colors)
    bar.set_facecolor(colors[idx])
    bar.set_hatch(hatches[idx])

ax.set_xlabel("Frequency")
ax.set_ylabel("POS Tag")
ax.set_title("Overall POS Tag Distribution")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.grid(axis="x", linestyle="--", alpha=0.7)
for bar, val in zip(bars, pos_counts.values[::-1]):
    ax.text(val + (pos_counts.max() * 0.005), bar.get_y() + bar.get_height()/2, f"{val:,}", va="center", fontsize=8, weight="bold")
out1 = f"{OUT_DIR}/01_distribusi_pos_tag.png"
fig.savefig(out1)
plt.close(fig)
print(f"  [Saved] {out1}")

# ────────────────────────────────────────────────────────────
# 5. CHART 2 – Main POS Categories
# ────────────────────────────────────────────────────────────
cat_counts = df["pos_main"].value_counts()
fig, ax = plt.subplots(figsize=(9, max(5, len(cat_counts) * 0.45)))
colors_cat = ["#333333", "#555555", "#777777", "#999999"]
hatches_cat = ["", "///", "\\\\", "xx"]
bars = ax.barh(cat_counts.index[::-1], cat_counts.values[::-1], color="#ffffff", edgecolor="black", linewidth=0.8)

for i, bar in enumerate(bars):
    idx = i % len(colors_cat)
    bar.set_facecolor(colors_cat[idx])
    bar.set_hatch(hatches_cat[idx])

ax.set_xlabel("Frequency")
ax.set_ylabel("Main Category")
ax.set_title("Main POS Category Distribution")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.grid(axis="x", linestyle="--", alpha=0.7)
for bar, val in zip(bars, cat_counts.values[::-1]):
    ax.text(val + (cat_counts.max() * 0.005), bar.get_y() + bar.get_height()/2, f"{val:,}", va="center", weight="bold")
out2 = f"{OUT_DIR}/02_kategori_pos_utama.png"
fig.savefig(out2)
plt.close(fig)
print(f"  [Saved] {out2}")

# ────────────────────────────────────────────────────────────
# 6. CHART 3 – Sentence length histogram + boxplot
# ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(tokens_per_sent.values, bins=30, color="#555555", edgecolor="#222222", linewidth=0.5)
axes[0].set_xlabel("Number of Tokens per Sentence")
axes[0].set_ylabel("Sentence Frequency")
axes[0].set_title("Sentence Length Distribution")
axes[0].axvline(tokens_per_sent.mean(), color="black", linestyle="--", label=f"Mean={tokens_per_sent.mean():.1f}")
axes[0].legend()
axes[0].grid(axis="y", linestyle="--")

axes[1].boxplot(
    tokens_per_sent.values, vert=True, patch_artist=True,
    boxprops=dict(facecolor="#cccccc", color="#000000"),
    medianprops=dict(color="#000000", linewidth=2),
    tick_labels=["Sentences"]
)
axes[1].set_ylabel("Number of Tokens per Sentence")
axes[1].set_title("Sentence Length Boxplot")
axes[1].grid(axis="y", linestyle="--")
out3 = f"{OUT_DIR}/03_panjang_kalimat.png"
fig.savefig(out3)
plt.close(fig)
print(f"  [Saved] {out3}")

# ────────────────────────────────────────────────────────────
# 7. CHART 4 – Top-30 Tokens
# ────────────────────────────────────────────────────────────
punc_filter = {",", ".", "?", "!", ":", ";", "(", ")", "*", "=", "-LRB-", "-RRB-", "\"", "'", "–", "—"}
token_filtered = df[~df["token"].isin(punc_filter) & (df["token"].str.len() > 1)]
top_tokens = token_filtered["token"].value_counts().head(30)
fig, ax = plt.subplots(figsize=(9, 9))
ax.barh(top_tokens.index[::-1], top_tokens.values[::-1], color="#444444", edgecolor="black", linewidth=0.7)
ax.set_xlabel("Frequency")
ax.set_title("Top-30 Most Frequent Tokens")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.grid(axis="x", linestyle="--", alpha=0.7)
for i, val in enumerate(top_tokens.values[::-1]):
    ax.text(val + 0.5, i, f"{val:,}", va="center", ha="left", fontsize=8)
out4 = f"{OUT_DIR}/04_top30_token.png"
fig.savefig(out4)
plt.close(fig)
print(f"  [Saved] {out4}")

# ────────────────────────────────────────────────────────────
# 8. CHART 5 – Heatmap POS per file
# ────────────────────────────────────────────────────────────
heatmap_data = pd.crosstab(df["source_file"], df["pos_main"], normalize="index")
fig, ax = plt.subplots(figsize=(14, 10))
im = ax.imshow(heatmap_data, cmap="Greys", aspect="auto")

# Add text annotations inside cells (like reference image 2)
for i in range(len(heatmap_data.index)):
    for j in range(len(heatmap_data.columns)):
        val = heatmap_data.iloc[i, j]
        if val > 0: # Only show non-zero to avoid clutter
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", 
                    color="white" if val > 0.4 else "black", fontsize=8)

ax.set_xticks(np.arange(len(heatmap_data.columns)))
ax.set_yticks(np.arange(len(heatmap_data.index)))
ax.set_xticklabels(heatmap_data.columns, rotation=45, ha="right")
ax.set_yticklabels(heatmap_data.index)
ax.set_title("POS Category Proportion per Source File")
plt.colorbar(im, label="Proportion")
plt.tight_layout()
out5 = f"{OUT_DIR}/05_heatmap_pos_per_file.png"
fig.savefig(out5)
plt.close(fig)
print(f"  [Saved] {out5}")

# ────────────────────────────────────────────────────────────
# 9. CHART 6 – Token length boxplot per category
# ────────────────────────────────────────────────────────────
pos_list = df["pos_main"].value_counts().index[:12]
plot_data = [df[df["pos_main"] == p]["token_len"] for p in pos_list]
fig, ax = plt.subplots(figsize=(12, 6))
ax.boxplot(plot_data, patch_artist=True, boxprops=dict(facecolor="#dddddd"), tick_labels=pos_list)
ax.set_ylabel("Token Length (chars)")
ax.set_title("Token Length Distribution per POS Category")
ax.grid(axis="y", linestyle="--")
out6 = f"{OUT_DIR}/06_panjang_token_per_pos.png"
fig.savefig(out6)
plt.close(fig)
print(f"  [Saved] {out6}")

# ────────────────────────────────────────────────────────────
# 10. CHART 7 – POS Category Pie Chart
# ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
main_top = cat_counts.head(7)
others_val = cat_counts.iloc[7:].sum()
pie_data = pd.concat([main_top, pd.Series({"Others": others_val})])
colors_pie = [plt.cm.Greys(i) for i in np.linspace(0.3, 0.8, len(pie_data))]
ax.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%", startangle=140, colors=colors_pie, wedgeprops={"edgecolor":"black"})
ax.set_title("Proportion of Main POS Categories")
out7 = f"{OUT_DIR}/07_pie_kategori_pos.png"
fig.savefig(out7)
plt.close(fig)
print(f"  [Saved] {out7}")

# ────────────────────────────────────────────────────────────
# 11. CHART 8 – Sentences per Source File
# ────────────────────────────────────────────────────────────
sent_per_file = df.groupby("source_file")["global_sentence_id"].nunique().sort_values()
fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(sent_per_file.index, sent_per_file.values, color="#ffffff", edgecolor="black", linewidth=0.8)

for i, bar in enumerate(bars):
    idx = i % len(colors) # Reuse colors from Chart 1
    bar.set_facecolor(colors[idx])
    bar.set_hatch(hatches[idx])

ax.set_xlabel("Number of Sentences")
ax.set_title("Number of Sentences per Source File")
ax.grid(axis="x", linestyle="--", alpha=0.7)
# Add value labels
for bar in bars:
    val = bar.get_width()
    ax.text(val + (sent_per_file.max() * 0.005), bar.get_y() + bar.get_height()/2, f"{int(val)}", va="center", fontsize=8)

out8 = f"{OUT_DIR}/08_kalimat_per_file.png"
fig.savefig(out8)
plt.close(fig)
print(f"  [Saved] {out8}")

# ────────────────────────────────────────────────────────────
# 12. CHART 9 – Top-10 & Bottom-5 POS Tags
# ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
bars_top = axes[0].barh(top10_pos.index[::-1], top10_pos.values[::-1], color="#ffffff", edgecolor="black", linewidth=0.8)
for i, bar in enumerate(bars_top):
    idx = i % len(colors)
    bar.set_facecolor(colors[idx])
    bar.set_hatch(hatches[idx])
axes[0].set_title("Top-10 Most Frequent POS Tags")
axes[0].grid(axis="x", linestyle="--")

bars_bot = axes[1].barh(bottom5_pos.index[::-1], bottom5_pos.values[::-1], color="#aaaaaa", edgecolor="black", linewidth=0.8, hatch="...")
axes[1].set_title("Bottom-5 Least Frequent POS Tags")
axes[1].grid(axis="x", linestyle="--")
out9 = f"{OUT_DIR}/09_top10_bottom5_pos.png"
fig.savefig(out9)
plt.close(fig)
print(f"  [Saved] {out9}")

# ────────────────────────────────────────────────────────────
# 13. CHART 10 – C. Noise Profile
# ────────────────────────────────────────────────────────────
# 1. Out-of-Vocabulary (OOV) Analysis
# Token is OOV if it's different from its formal KBBI version
df["is_oov"] = df["token"].str.lower().str.strip() != df["KBBI"].astype(str).str.lower().str.strip()
n_oov = int(df["is_oov"].sum())
pct_oov = n_oov / n_tokens * 100

# 2. Character elongation: any character repeated 3+ consecutive times
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    df["is_elongated"] = df["token"].str.contains(r"(.)\1{2,}", regex=True, na=False)
n_elongated  = int(df["is_elongated"].sum())
pct_elongated = n_elongated / n_tokens * 100

# 3. Informal NG-prefix (nasalization): token starts with 'ng' + letter
df["has_ng_prefix"] = df["token"].str.match(r"(?i)^ng[a-z]", na=False)
n_ng  = int(df["has_ng_prefix"].sum())
pct_ng = n_ng / n_tokens * 100

# Total noisy tokens (union of OOV, elongated, and ng-prefix)
df["is_noisy"] = df["is_oov"] | df["is_elongated"] | df["has_ng_prefix"]
n_noisy  = int(df["is_noisy"].sum())
n_clean  = n_tokens - n_noisy

print(f"\n{'─'*60}")
print("  C. NOISE PROFILE (Highly Noisy Text Analysis)")
print(f"{'─'*60}")
print(f"  OOV Tokens (vs KBBI)  : {n_oov:,} tokens ({pct_oov:.2f}%)")
print(f"  Character Elongation  : {n_elongated:,} tokens ({pct_elongated:.2f}%)")
print(f"  Informal NG-Prefix    : {n_ng:,} tokens ({pct_ng:.2f}%)")
print(f"  Total Unique Noisy    : {n_noisy:,} tokens ({n_noisy/n_tokens*100:.2f}%)")
print(f"  Non-noisy (Pure IV)   : {n_clean:,} tokens ({n_clean/n_tokens*100:.2f}%)")

# Chart 10: Noise breakdown
categories = ["OOV (vs KBBI)", "Elongated", "NG-Prefix", "Non-noisy"]
counts_n   = [n_oov, n_elongated, n_ng, n_clean]
colors_n   = ["#333333", "#666666", "#999999", "#cccccc"]
hatches_n  = ["", "///", "\\\\", "xx"]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(categories, counts_n, color="#ffffff", edgecolor="black", linewidth=1.0)
for i, bar in enumerate(bars):
    bar.set_facecolor(colors_n[i])
    bar.set_hatch(hatches_n[i])

ax.set_ylabel("Number of Tokens")
ax.set_title("Text Noise Profile Analysis")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.grid(axis="y", linestyle="--", alpha=0.7)

for bar, cnt in zip(bars, counts_n):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + n_tokens * 0.01,
        f"{cnt:,}\n({cnt/n_tokens*100:.2f}%)",
        ha="center", va="bottom", fontsize=9, weight="bold"
    )

plt.tight_layout()
out10 = f"{OUT_DIR}/10_noise_profile.png"
fig.savefig(out10)
plt.close(fig)
print(f"  [Saved] {out10}")

# ────────────────────────────────────────────────────────────
# 14. FINAL SUMMARY
# ────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("  LIST OF OUTPUT FILES")
print(f"{'─'*60}")
outputs = [(out1, "POS tag labels"), (out2, "Main POS categories"), (out3, "Sentence length"), (out4, "Top-30 tokens"), (out5, "Heatmap"), (out6, "Token length"), (out7, "Pie chart"), (out8, "Sentences/file"), (out9, "Top-10/Bottom-5"), (out10, "Noise profile")]
for path, desc in outputs:
    print(f"  [✓] {path:<45}  {desc}")
print(f"\n  All outputs saved in: ./{OUT_DIR}/")
print("=" * 60)
print("  DONE")
print("=" * 60)
