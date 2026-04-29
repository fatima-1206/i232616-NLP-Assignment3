# Assignment 3: Transformer-based Review Understanding with RAG Enhanced Explanation Generation

This repository contains a Jupyter notebook implementation of a three-stage NLP pipeline for Amazon review understanding:

- Part 0: preprocessing and dataset preparation
- Part A: encoder-only Transformer for multi-task review understanding
- Part B: embedding-based retrieval over training reviews
- Part C: decoder-only Transformer for explanation generation with retrieval-augmented context

## Quick Start

1. Open `i232616-NLP-Assignment3.ipynb` in VS Code or Jupyter.
2. Run the notebook from top to bottom.
3. Make sure the dataset files are available under `Dataset/`.
4. Ensure the output folders `models/` and `results/` are writable.

The notebook is designed to generate all required artifacts, including model checkpoints, embeddings, metrics, plots, and qualitative outputs.

## Repository Structure

- `i232616-NLP-Assignment3.ipynb` - main notebook for the full pipeline
- `i232616.ipynb` - earlier notebook version / working draft
- `Dataset/` - Amazon review data subsets used by the notebook
- `models/` - saved trained model weights
- `results/` - evaluation files, embeddings, plots, and configuration snapshots
- `df.csv` - exported combined dataset snapshot

## Project Requirements Covered

### Part 0: Preprocessing

The notebook performs the following steps using only the training split for vocabulary construction:

- data loading from three product categories
- sentiment label mapping:
  - 1-2 stars -> Negative
  - 3 stars -> Neutral
  - 4-5 stars -> Positive
- derived feature creation using review length categories:
  - Short
  - Medium
  - Long
- text cleaning
- tokenization
- vocabulary building from training data only
- token-to-index conversion
- padding and truncation to a fixed maximum sequence length
- saving preprocessing configuration and serialized datasets

### Part A: Encoder Model

The encoder is implemented from scratch and includes:

- token embeddings and sinusoidal positional encodings
- custom scaled dot-product attention
- custom multi-head self-attention
- stacked encoder blocks with residual connections and layer normalization
- multi-task heads for sentiment classification and length-category prediction
- pooled review embeddings exported for retrieval

### Part B: Retrieval Module

The retrieval stage uses the encoder embeddings to:

- store training review vectors on disk
- normalize embeddings for cosine similarity search
- retrieve top-k similar training reviews for a query review
- analyze retrieval quality with similarity and label-match statistics
- save retrieval configuration for downstream use

### Part C: Decoder and RAG Pipeline

The decoder stage includes:

- decoder-only Transformer implemented from scratch
- causal masking to prevent attention to future tokens
- autoregressive generation
- prompt construction from the review, predicted labels, and retrieved neighbors
- perplexity evaluation on the test set
- qualitative generation examples
- ablation study comparing full RAG vs no-retrieval baseline

## Expected Outputs

Running the notebook should produce files such as:

- `models/partA_best_encoder.pt`
- `models/partC_best_decoder.pt`
- `results/partA_train_embeddings.npy`
- `results/partA_val_embeddings.npy`
- `results/partA_test_embeddings.npy`
- `results/partA_metrics.json`
- `results/preprocessing_config.json`
- `results/retrieval_config.json`
- `results/decoder_evaluation_baseline.json`
- `results/partC_qualitative_examples.json`
- `results/partC_rag_ablation_study.json`
- `results/*.png` plots and visualizations

## Requirements

The notebook was built to satisfy the assignment constraints:

- Transformer components are implemented manually
- pretrained Transformer models are not used
- `nn.Transformer`, `nn.MultiheadAttention`, and `nn.TransformerEncoder` are not used
- preprocessing is done manually
- the encoder is trained as a multi-task model
- retrieval uses encoder embeddings from Part A
- the decoder is trained autoregressively with a causal mask

## Recommended Environment

The notebook expects a Python environment with at least the following packages:

- `pandas`
- `numpy`
- `torch`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tqdm`

If you are running locally, install dependencies before executing the notebook.

## Notes on Execution

- Run the notebook in order; later cells depend on artifacts produced earlier.
- If checkpoints are missing, enable the training flags in the notebook and rerun the relevant sections.
- The notebook saves intermediate files in `results/` so later stages can reuse them without recomputation.

## Report Deliverables

The assignment report should include:

- overall system design and methodology
- preprocessing pipeline description
- justification of design choices for each part
- evaluation results and plots
- hyperparameter tuning analysis
- RAG ablation study
- discussion of retrieval quality and generation quality

## Troubleshooting

- If Part A evaluation fails, verify that the encoder checkpoint exists in `models/`.
- If Part C evaluation fails, verify that the decoder checkpoint exists in `models/`.
- If retrieval files are missing, rerun Part A so embeddings and metadata are exported to `results/`.
- If the dataset cannot be loaded, confirm that the files in `Dataset/` are present and readable.

## Author

Fatima Tu Zahra
23i-2616
BDS-6B
Spring 2026
