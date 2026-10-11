import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import torch, matplotlib.pyplot as plt, numpy as np
from src.scorer import load_scorer
from src.compressor import compress

st.set_page_config(page_title="Prompt Compressor", layout="wide")
st.title("Learned Prompt Compression")
st.caption("DistilBERT token scorer · research analytics")

@st.cache_resource
def get_scorer():
    models_path = os.path.join(os.path.dirname(__file__), "..", "models", "scorer.pt")
    return load_scorer(os.path.abspath(models_path))

scorer = get_scorer()

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    ratio = st.slider("Compression ratio", 0.1, 0.9, 0.5, 0.05,
                      help="Fraction of tokens to keep")
    show_heatmap = st.checkbox("Show token salience heatmap", value=True)

# ── Input ─────────────────────────────────────────────────────────────────────
prompt = st.text_area("Input prompt", height=180,
    placeholder="Paste any text here to compress it...")

if st.button("Compress", type="primary") and prompt.strip():
    result = compress(prompt, scorer, ratio=ratio)

    # ── Top metrics row ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Original tokens",   result["original_len"])
    col2.metric("Compressed tokens", result["compressed_len"])
    col3.metric("Token savings",     f"{result['savings_pct']}%")
    col4.metric("Ratio kept",        f"{ratio:.0%}")

    # ── Compressed output ─────────────────────────────────────────────────────
    st.subheader("Compressed prompt")
    st.info(result["compressed"])

    # ── Token salience heatmap ────────────────────────────────────────────────
    if show_heatmap:
        st.subheader("Token salience scores")
        tokens = result["tokens"]
        scores = np.array(result["scores"])
        kept   = result["kept_mask"]

        fig, ax = plt.subplots(figsize=(14, 1.8))
        cmap = plt.get_cmap("RdYlGn")
        for i, (t, s, k) in enumerate(zip(tokens, scores, kept)):
            ax.add_patch(plt.Rectangle((i, 0), 0.9, 1,
                         color=cmap(s), alpha=0.85))
            ax.text(i+0.45, 0.5, t.replace("▁","").replace("Ġ"," "),
                    ha="center", va="center", fontsize=7.5,
                    color="black" if s > 0.35 else "white")
            if not k:
                ax.add_patch(plt.Rectangle((i, 0), 0.9, 1,
                             fill=False, edgecolor="#cc3333", linewidth=1.5))
        ax.set_xlim(0, len(tokens)); ax.set_ylim(0, 1); ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        st.caption("Green = high importance  ·  Red border = dropped token")

    # ── Score distribution ─────────────────────────────────────────────────────
    st.subheader("Score distribution")
    fig2, ax2 = plt.subplots(figsize=(6, 2.5))
    ax2.hist(result["scores"], bins=30, color="#1D9E75", alpha=0.8)
    threshold = sorted(result["scores"], reverse=True)[result["compressed_len"]-1]
    ax2.axvline(threshold, color="#D85A30", linewidth=1.5, linestyle="--",
                label=f"Threshold @ ratio={ratio}")
    ax2.set_xlabel("Score"); ax2.set_ylabel("Count"); ax2.legend()
    st.pyplot(fig2, use_container_width=True)