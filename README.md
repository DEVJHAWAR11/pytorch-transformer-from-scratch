# Transformer From Scratch

A full encoder-decoder Transformer implemented from scratch in PyTorch.
Based on the paper **"Attention is All You Need"** (Vaswani et al., 2017).

## Task
Machine Translation: English → German using the Multi30K dataset.

## Architecture
- Scaled Dot-Product Attention
- Multi-Head Attention
- Positional Encoding
- Position-wise Feed Forward Network
- Encoder Stack (6 layers)
- Decoder Stack (6 layers)
- Noam LR Scheduler + Label Smoothing

## Project Structure
- `src/` — all model modules
- `notebooks/` — step-by-step Colab notebooks
- `data/` — dataset loading instructions
