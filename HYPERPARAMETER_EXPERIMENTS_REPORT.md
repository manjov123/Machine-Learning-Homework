# Hyperparameter Experiments Report
## CS4375 Assignment 2 - Sentiment Analysis

I ran 11 experiments to compare FFNNs and RNNs for predicting restaurant ratings from review text. Bottom line: FFNN works much better here.

## Quick Summary

- FFNN best: 60.25% accuracy (hidden_dim=20, epochs=5)
- RNN best: 36.25% accuracy (hidden_dim=16, epochs=5) 
- FFNN wins by 24 percentage points
- RNN overfits almost immediately (1-2 epochs)

## Dataset

- 16,000 training reviews
- 800 validation reviews
- 800 test reviews
- 5 classes (1-5 stars, encoded as 0-4)

## Models

**FFNN:** Bag-of-words → Hidden Layer (ReLU) → Output  
**RNN:** Word Embeddings (50-dim) → RNN (tanh) → Output

## FFNN Experiments

### Varying Hidden Dimensions (5 epochs)

| Hidden Dim | Epoch 1 Val | Epoch 3 Val | Epoch 5 Val | Best Val Acc |
|------------|-------------|-------------|-------------|--------------|
| 5          | 35%         | 56.4%       | 54.8%       | 56.4% (E3)   |
| 10         | 31.5%       | 45%         | 50%         | 50% (E5)     |
| **20**     | **36%**     | **57.8%**   | **60.3%**   | **60.3% (E5)** |
| 50         | 43%         | 57.9%       | 49.9%       | 57.9% (E3)   |
| 100        | 42%         | 57.6%       | 55.9%       | 57.6% (E3)   |

The sweet spot is 20 hidden neurons. Smaller dimensions underfit, larger ones overfit around epoch 3. Medium-sized hidden layers (20-50) work best.

### Varying Epochs (hidden_dim=20)

| Epochs | Best Val Acc | Best Epoch | Notes |
|--------|--------------|------------|-------|
| 5      | 60.25%       | 5          | Optimal |
| 10     | 60.00%       | 6          | Starts overfitting after E6 |
| 15     | 60.25%       | 5          | Heavy overfitting (76.5% train, 56.8% val at E15) |

Training for 5-6 epochs is enough. Beyond that, the model memorizes training data but validation accuracy drops.

## RNN Experiments

### Varying Hidden Dimensions

| Hidden Dim | Epoch 1 Val | Epoch 2 Val | Best Val Acc | Stopped At |
|------------|-------------|-------------|--------------|------------|
| **16**     | **36.3%**   | **15.1%**   | **36.3% (E1)** | 2 epochs   |
| 32         | 29.5%       | 16.5%       | 29.5% (E1)   | 2 epochs   |
| 64         | 19.8%       | 19.6%       | 19.8% (E1)   | 2 epochs   |
| 128        | 25.8%       | 27.4%       | 27.4% (E2)   | 3 epochs   |

The RNN struggles regardless of architecture. It peaks at epoch 1 and crashes by epoch 2. Early stopping kicks in almost immediately for every configuration.

## Why FFNN Wins

1. **Task characteristics**: For sentiment, specific words matter more than their order. "Amazing" indicates a positive review whether it's at the beginning or end.

2. **Training stability**: FFNN trains smoothly over 5 epochs. RNN collapses after 1 epoch.

3. **Data efficiency**: 16,000 examples is plenty for FFNN but not enough for RNN to learn sequential patterns.

4. **Model complexity**: RNN has more parameters but can't use them effectively on this dataset. It overfits before learning anything useful.

Interesting quirk: The RNN's training accuracy (25%) is actually lower than its validation accuracy (36%), suggesting it's not even fitting the training data properly. This points to optimization issues with the vanilla RNN architecture.

## Learning Curves

**FFNN progression:**
- Epoch 1: 40% train, 36% val
- Epoch 3: 54% train, 58% val (nice generalization)
- Epoch 5: 60% train, 60% val (optimal)
- Epoch 10: 69% train, 50% val (overfitting)

**RNN progression:**
- Epoch 1: 25% train, 36% val (already problematic)
- Epoch 2: 26% train, 15% val (collapsed)
- Early stopping triggered

## Complete Results

### FFNN Summary

| Experiment | Hidden Dim | Epochs | Best Val Acc | Overfitting? |
|------------|------------|--------|--------------|--------------|
| 1          | 5          | 5      | 56.38%       | No           |
| **2**      | **20**     | **5**  | **60.25%**   | **No**       |
| 3          | 50         | 5      | 57.88%       | Slight       |
| 4          | 100        | 5      | 57.63%       | Slight       |
| 5          | 20         | 10     | 60.00%       | Yes          |
| 6          | 20         | 15     | 60.25%       | Severe       |

### RNN Summary

| Experiment | Hidden Dim | Max Epochs | Stopped At | Best Val Acc |
|------------|------------|------------|------------|--------------|
| **1**      | **16**     | 5          | 2          | **36.25%**   |
| 2          | 32         | 5          | 2          | 29.50%       |
| 3          | 64         | 10         | 2          | 19.75%       |
| 4          | 128        | 10         | 3          | 27.38%       |
| 5          | 32         | 15         | 2          | 29.50%       |

## Recommendations

### For FFNN
The current configuration (hidden_dim=20, epochs=5) is already near-optimal for this dataset. If I were to try improvements:
- Experiment with Adam optimizer instead of SGD
- Add dropout (0.2-0.3) to push past 60% without overfitting
- Try learning rate scheduling

### For RNN
The vanilla RNN architecture isn't working. To fix it:
- Switch to LSTM or GRU (better gradient flow)
- Add dropout (0.5) throughout the network
- Lower learning rate to 0.001
- Use gradient clipping to prevent exploding gradients
- Fine-tune the word embeddings instead of keeping them frozen
- Try bidirectional processing

Or honestly, just use FFNN for this task. The RNN would need significant work to match even 50% accuracy.

## Practical Takeaway

Use FFNN (hidden_dim=20, epochs=5) for restaurant review sentiment analysis. It's:
- 3× better than random guessing
- 66% better than RNN
- Fast (~10 seconds per epoch)
- Simple to implement and tune

The bag-of-words approach works because sentiment is mostly about which words appear, not their sequence. Reviews with "terrible", "awful", "worst" are negative regardless of grammar or structure.

## Reproduction

Best FFNN:
```bash
python3 ffnn.py --hidden_dim 20 --epochs 5 \
  --train_data ./training.json \
  --val_data ./validation.json \
  --test_data ./test.json
```

Best RNN:
```bash
python3 rnn.py --hidden_dim 16 --epochs 5 \
  --train_data training.json \
  --val_data validation.json \
  --test_data test.json
```

---

March 11, 2026  
11 experiments (6 FFNN + 5 RNN)  
Best model: FFNN (hidden_dim=20, epochs=5) - 60.25% validation accuracy
