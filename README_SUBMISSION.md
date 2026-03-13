# CS4375 Assignment 2 - Sentiment Analysis
## Neural Networks for Restaurant Review Rating Prediction

**Manish Nadendla**  
March 10, 2026

## Overview

I implemented and compared two neural networks for predicting restaurant ratings (1-5 stars) from review text:

- **FFNN**: Bag-of-words approach
- **RNN**: Sequential word embeddings approach

**Spoiler**: FFNN wins by a lot (60.25% vs 36.25%).

**Dataset**: 16,000 training reviews, 800 validation, 800 test

## Implementation

### FFNN (`ffnn.py`)

Simple architecture: bag-of-words input → hidden layer (ReLU) → 5-class output (LogSoftmax).

The key part I implemented:
```python
def forward(self, input_vector):
    hidden_layer = self.activation(self.W1(input_vector))
    output_layer = self.W2(hidden_layer)
    predicted_vector = self.softmax(output_layer)
    return predicted_vector
```

### RNN (`rnn.py`)

Uses word embeddings: sequence input → RNN (tanh) → 5-class output.

The forward pass:
```python
def forward(self, inputs):
    _, hidden = self.rnn(inputs)
    output = self.W(hidden[-1])
    output = output.view(-1)
    predicted_vector = self.softmax(output.view(1, -1))
    return predicted_vector
```

Both models output predictions to `results/test.out` (800 lines, values 0-4).

## How to Run

**Best FFNN:**
```bash
python ffnn.py --hidden_dim 20 --epochs 5 \
  --train_data ./training.json \
  --val_data ./validation.json \
  --test_data ./test.json
```

**Best RNN:**
```bash
python rnn.py --hidden_dim 16 --epochs 5 \
  --train_data training.json \
  --val_data validation.json \
  --test_data test.json
```

## Experiments

Ran 11 experiments total (6 FFNN, 5 RNN) testing different hidden dimensions and epochs.

### FFNN Results

| Hidden Dim | Epochs | Best Val Acc | Notes |
|------------|--------|--------------|-------|
| 5          | 5      | 56.4%        | Too small |
| **20**     | **5**  | **60.25%**   | **Optimal** |
| 50         | 5      | 57.9%        | Overfits at E3 |
| 100        | 5      | 57.6%        | Overfits at E3 |
| 20         | 10     | 60.0%        | Overfits after E6 |
| 20         | 15     | 60.25%       | Heavy overfitting |

**Winner**: hidden_dim=20, epochs=5 → 60.25% validation accuracy

Training is stable. The model learns gradually over 5 epochs, hitting peak performance right when training and validation accuracies align (both ~60%). Beyond 6 epochs, it starts memorizing the training data.

### RNN Results

| Hidden Dim | Max Epochs | Stopped At | Best Val Acc | Notes |
|------------|------------|------------|--------------|-------|
| **16**     | 5          | 2          | **36.25%**   | Best performance |
| 32         | 5          | 2          | 29.5%        | Quick overfit |
| 64         | 10         | 2          | 19.75%       | Poor |
| 128        | 10         | 3          | 27.4%        | Quick overfit |

**Winner**: hidden_dim=16, epochs=5 → 36.25% validation accuracy (stopped at epoch 2)

The RNN struggles. It peaks at epoch 1 then collapses by epoch 2, triggering early stopping immediately. The training accuracy (25%) is actually lower than validation (36%), which suggests it's not even fitting the training data properly.

## Why FFNN Wins

| Metric | FFNN | RNN | Difference |
|--------|------|-----|------------|
| Validation Accuracy | 60.25% | 36.25% | +24% |
| Training Accuracy | 60.24% | 25.04% | +35% |
| Training Time/Epoch | ~10 sec | ~50 sec | 5× faster |
| Training Stability | Stable | Crashes after E1 | - |

**The fundamental reason**: For sentiment analysis, specific words matter more than their order. "This place is amazing" and "Amazing place, this is" have the same sentiment. The bag-of-words representation captures this naturally.

The RNN has three main problems:
1. **Overfitting**: Crashes after 1 epoch regardless of architecture
2. **Vanishing gradients**: Vanilla RNN can't handle long sequences well
3. **Data hungry**: 16,000 examples isn't enough for it to learn sequential patterns

FFNN just needs to learn "amazing" = good, "terrible" = bad. RNN tries to learn grammar and context, which is overkill here.

### Learning Curves

**FFNN** (hidden_dim=20):
- Epoch 1: 40% train, 36% val (learning)
- Epoch 3: 54% train, 58% val (good generalization)
- Epoch 5: 60% train, 60% val (optimal)
- Epoch 10: 69% train, 50% val (overfitting)

**RNN** (hidden_dim=16):
- Epoch 1: 25% train, 36% val (not learning properly)
- Epoch 2: 26% train, 15% val (collapsed)

## What I'd Try Next

**For FFNN** (already pretty good):
- Try Adam optimizer instead of SGD
- Add dropout (0.2-0.3) to push past 60%
- Experiment with learning rate scheduling

**For RNN** (needs major work):
- Switch to LSTM or GRU instead of vanilla RNN
- Add dropout (0.5) throughout
- Lower learning rate to 0.001
- Use gradient clipping
- Fine-tune the word embeddings
- Or just use FFNN—it's working great

## Bottom Line

Use FFNN (hidden_dim=20, epochs=5) for this task. It's:
- 3× better than random guessing (20% baseline)
- 66% better than RNN
- Fast to train (~50 seconds total)
- Simple and stable

The RNN would need significant architectural improvements (LSTM, dropout, careful tuning) to even reach 50% accuracy. Not worth it for this problem.

## Files

**Code**: `ffnn.py`, `rnn.py`  
**Data**: `training.json`, `validation.json`, `test.json`, `word_embedding.pkl`  
**Output**: `results/test.out` (800 predictions, 0-4)

## Technical Notes

**Dependencies**: Python 3.x, PyTorch, NumPy, tqdm

**FFNN specs**: ~50k vocab size, 20 hidden neurons, 5 outputs (~1M parameters), SGD optimizer (lr=0.01, momentum=0.9)

**RNN specs**: 50-dim embeddings, 16 hidden neurons, 5 outputs (~1.2k parameters), Adam optimizer (lr=0.01)

Training completed on CPU, no GPU needed.

---

March 11, 2026  
Best model: FFNN (hidden_dim=20, epochs=5) - 60.25% validation accuracy
