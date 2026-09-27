# Tweet Sentiment Analysis — TF-IDF + PyTorch LSTM

A sentiment classification pipeline for tweets, exploring TF-IDF feature extraction alongside a PyTorch `Embedding + LSTM` classifier trained on GPU.

## Pipeline

1. **Load data** — reads `tweet_sentiment.csv` (columns: `cleaned_tweets`, `sentiment`).
2. **Handle missing values** — fills missing `cleaned_tweets` with the mode.
3. **Tokenize & stem** — uses NLTK's `word_tokenize` and `PorterStemmer`.
4. **TF-IDF exploration** — vectorizes stemmed text with `TfidfVectorizer` (max 5000 features) to inspect feature space.
5. **Label encoding** — encodes sentiment labels with `LabelEncoder`, and separately remaps `-1 → 2` for the LSTM's 3-class output.
6. **Custom tokenizer & padding** — a lightweight `SimpleTokenizer` builds a frequency-sorted vocabulary and pads/truncates sequences to length 100 (implemented from scratch, no external dependency).
7. **Model** — PyTorch `nn.Embedding → nn.LSTM(64) → Dropout(0.3) → Linear(3)`, trained with Adam + cross-entropy loss, with manual early stopping (patience=3) on validation loss and best-weight restoration.
8. **Evaluation** — reports test accuracy on the held-out split.

## Requirements

See `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

You'll also need the NLTK `punkt_tab` tokenizer data (downloaded automatically in the notebook via `nltk.download('punkt_tab')`).

A CUDA-capable GPU is optional but recommended — the notebook automatically falls back to CPU if none is available (`torch.cuda.is_available()`).

## Usage

1. Place your `tweet_sentiment.csv` file in the project root (or update the path in the notebook — it checks the current folder, then `/content/` for Colab).
2. Open and run `sentiment_tfidf_pytorch_lstm.ipynb` in Jupyter, Colab, or VS Code.

## Data

The dataset is not included in this repository. It should have at least two columns:

| column | description |
|---|---|
| `cleaned_tweets` | preprocessed tweet text |
| `sentiment` | sentiment label (`-1`, `0`, `1`) |

## Notes

- This is a PyTorch-based companion to a separate Keras/TensorFlow version of the same task.
- The TF-IDF vectorization is exploratory (shape/inspection only) and is not the input to the LSTM model — the LSTM uses its own learned embeddings via the custom tokenizer.
- Feel free to tune embedding size, LSTM hidden size, dropout, batch size, or number of epochs.
