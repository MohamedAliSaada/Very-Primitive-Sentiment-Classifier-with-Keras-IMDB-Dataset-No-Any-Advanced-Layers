# Sentiment Classifier with Keras & IMDB Dataset

This project builds a binary sentiment classification model using the IMDB dataset from Hugging Face and Keras Sequential API with only Dense layers and embeddings.
 
---

## 📦 Dataset

The data is loaded directly from Hugging Face:
- Dataset: `stanfordnlp/imdb`
- Format: `.parquet` files
- Splits used: `train`, `test`

---

## 🧠 Model Architecture

- `Embedding(input_dim=90000, output_dim=100, input_length=500)`
- `GlobalAveragePooling1D()`
- `Dense(128, activation='relu')`
- `Dense(64, activation='relu')`
- `Dense(32, activation='relu')`
- `Dense(8, activation='relu')`
- `Dense(1, activation='sigmoid')`

> Optimizer: `adamax`  
> Loss: `binary_crossentropy`  
> Metric: `accuracy`

---

## 🏋️ Training Results

Training was run for **5 epochs** with `validation_split=0.25`.  
Here are the final results:

- ✅ Final Accuracy: `0.9385`
- ✅ Final Validation Accuracy: `0.8232`
- ❌ Final Validation Loss: `0.4833`

### 📈 Accuracy Plot

![Accuracy over Epochs](Accuracy%20over%20Epochs.png)

### 📉 Loss Plot

![Loss over Epochs](Loss%20over%20Epochs.png)

---

## 🧪 Test Evaluation

After evaluating the model on the test set:

- **Test Accuracy:** `0.9125`

![Test Accuracy](test%20acuracy.png)

---

## 💾 Saved Models

You can find the trained model files:
- `imdb.keras` – Keras format
- `imdb.h5` – HDF5 legacy format

---

## ▶️ How to Use

1. Load tokenizer and pad your input
2. Load the model using:
   ```python
   from keras.models import load_model
   model = load_model("imdb.h5")  # or imdb.keras
