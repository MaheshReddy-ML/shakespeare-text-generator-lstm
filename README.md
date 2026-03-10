# Shakespeare Text Generator (BiLSTM)

A deep learning based text generation model trained on Shakespeare dialogue.
The model learns language patterns using **Bidirectional LSTM networks** and generates Shakespeare-style text by predicting the next word in a sequence.

---

## Project Overview

This project demonstrates how **Recurrent Neural Networks (RNNs)** and **LSTM architectures** can be used to build a simple **language model**.
The model is trained on Shakespeare dialogue and learns to generate text in a similar style.

The training pipeline includes:

* Text preprocessing
* Tokenization
* N-gram sequence generation
* Sequence padding
* Bidirectional LSTM model training
* Text generation using a trained model

---

## Model Architecture

### Embedding Layer

↓ Bidirectional LSTM (256 units)

↓ Dropout (0.2)

↓ Bidirectional LSTM (128 units)

↓ Dropout (0.2)

↓ Dense Softmax Layer (Vocabulary Prediction)

Loss Function: `Sparse Categorical Crossentropy`
Optimizer: `Adam`

---

## Dataset

The model is trained on a **Shakespeare dialogue dataset** containing lines spoken by characters across multiple plays.

Dataset preprocessing includes:

* Removing missing values
* Filtering very short lines
* Converting text into token sequences
* Creating N-gram training samples

Approximate dataset statistics:

* Total lines: ~98,000
* Vocabulary size: ~14,000 words
* Generated training sequences: ~680,000

---

## Training Configuration

Epochs: 60
Batch Size: 128

Callbacks used:

* Early Stopping
* Reduce Learning Rate on Plateau

Approximate training results:

* Training Accuracy: ~25% – 30%
* Training Time: ~30 minutes per epoch (CPU)

Note: Accuracy appears low because the model predicts the next word from a large vocabulary (~14k words).

---

## Example Text Generation

Input Seed:

```
love
```

Example Generated Output:

```
love is the shadow of a noble heart that speaks
against the sorrow of the night
```

---

## Installation

Clone the repository:

```
git clone https://github.com/MaheshReddy-ML/shakespeare-text-generator-lstm.git
cd shakespeare-text-generator-lstm
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the Model

Run the training script:

```
python shakespeare_model.py
```

After training, the script allows you to generate Shakespeare-style text interactively.

Example:

```
Enter theme: love
Generated Line:
love shall be the light that guides thy noble heart
```

---

## Requirements

Main dependencies:

* TensorFlow
* Pandas
* NumPy

Install using:

```
pip install -r requirements.txt
```

---

## Note

The trained model file (`.keras`) and tokenizer (`.pkl`) are not included in the repository since they are generated after training.

Running the training script will recreate them.

---

## Purpose of the Project

This project was built as a learning exercise to understand:

* NLP preprocessing pipelines
* Sequence modeling
* LSTM based language models
* Neural text generation

It serves as a foundational step before exploring more advanced **Transformer-based language models**.
