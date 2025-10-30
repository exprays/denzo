# 🔄 AMR System Process Flow

This document explains the complete workflow and algorithm execution flow of the Automatic Modulation Recognition system.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Signal Generation Flow](#signal-generation-flow)
3. [Training Process Flow](#training-process-flow)
4. [Classification Process Flow](#classification-process-flow)
5. [Algorithm Details](#algorithm-details)

---

## System Overview

The AMR system operates in three main phases:

```
Phase 1: SIGNAL GENERATION
         ↓
Phase 2: CLASSIFIER TRAINING (one-time setup)
         ↓
Phase 3: SIGNAL CLASSIFICATION (repeated)
```

---

## Signal Generation Flow

### Step-by-Step Process

```
User Input Parameters
├── Modulation Type: BPSK/QPSK/QAM/16-QAM/64-QAM
├── Number of Samples: 512-4096
├── SNR (dB): 0-20
└── Channel Type: AWGN/Rayleigh
         ↓
┌────────────────────────────────────────┐
│  1. Symbol Generation                  │
│  ────────────────────────              │
│  • Random bit/symbol selection         │
│  • Map to constellation points         │
│  • Create complex baseband signal      │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  2. Channel Effects                    │
│  ────────────────────────             │
│  IF AWGN:                              │
│    • Calculate signal power            │
│    • Compute noise power from SNR      │
│    • Add complex Gaussian noise        │
│  IF Rayleigh:                          │
│    • Generate fading coefficients      │
│    • Apply fading to signal            │
│    • Add Gaussian noise                │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  3. Data Formatting                    │
│  ────────────────────────             │
│  • Convert numpy complex to dict       │
│  • Format: {real: float, imag: float}  │
│  • Limit to first 500 samples for UI   │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  4. Send to Frontend                   │
│  ────────────────────────             │
│  • JSON response with signal data      │
│  • Frontend renders visualizations     │
└────────────────────────────────────────┘
```

### Detailed Symbol Generation

#### BPSK (Binary Phase Shift Keying)

```python
Input: num_samples = 1024
Step 1: Generate random bits → [0, 1, 1, 0, 1, ...]
Step 2: Map to constellation → [-1, +1, +1, -1, +1, ...]
Output: Complex signal → [-1+0j, +1+0j, +1+0j, -1+0j, ...]

Constellation Points:
  +1 ────●

  -1 ●────
```

#### QPSK (Quadrature Phase Shift Keying)

```python
Input: num_samples = 1024
Step 1: Generate random symbols (0-3) → [2, 0, 3, 1, ...]
Step 2: Map to constellation:
  Symbol 0 → (+1+1j)/√2 =  0.707 + 0.707j  (45°)
  Symbol 1 → (-1+1j)/√2 = -0.707 + 0.707j  (135°)
  Symbol 2 → (+1-1j)/√2 =  0.707 - 0.707j  (-45°)
  Symbol 3 → (-1-1j)/√2 = -0.707 - 0.707j  (-135°)
Output: Complex signal with 4 unique points

Constellation:
     Q
     ↑
  ●  |  ●     (45° and 135°)
─────┼─────→ I
  ●  |  ●     (-45° and -135°)
```

#### 16-QAM (16-Quadrature Amplitude Modulation)

```python
Input: num_samples = 1024
Step 1: Generate random symbols (0-15) → [5, 12, 0, 9, ...]
Step 2: Create 4×4 grid constellation:
  I-values: [-3, -1, +1, +3]
  Q-values: [-3, -1, +1, +3]
  Normalize by √10
Output: 16 unique constellation points

Constellation (4×4 grid):
     Q
     ↑
  ●●●●
  ●●●●
─────┼─────→ I
  ●●●●
  ●●●●
```

#### 64-QAM

```python
Input: num_samples = 1024
Step 1: Generate random symbols (0-63)
Step 2: Create 8×8 grid constellation:
  I-values: [-7, -5, -3, -1, +1, +3, +5, +7]
  Q-values: [-7, -5, -3, -1, +1, +3, +5, +7]
  Normalize by √42
Output: 64 unique constellation points
```

### Channel Model Details

#### AWGN (Additive White Gaussian Noise)

```python
# Mathematical Model
received_signal = transmitted_signal + noise

# Implementation
signal_power = mean(|signal|²)
SNR_linear = 10^(SNR_dB / 10)
noise_power = signal_power / SNR_linear
noise_std = sqrt(noise_power / 2)

noise_real = random_normal(0, noise_std, num_samples)
noise_imag = random_normal(0, noise_std, num_samples)
noise = noise_real + 1j * noise_imag

output = signal + noise
```

**Effect on Signal:**

- Adds random perturbations to constellation points
- Points spread around original positions
- Higher SNR → tighter clusters
- Lower SNR → more spread

#### Rayleigh Fading Channel

```python
# Mathematical Model
received_signal = h * transmitted_signal + noise

# Implementation
# 1. Generate fading coefficients
h_real = random_normal(0, 1/√2, num_samples)
h_imag = random_normal(0, 1/√2, num_samples)
h = h_real + 1j * h_imag

# 2. Apply fading
faded_signal = h * signal

# 3. Add AWGN
output = add_awgn(faded_signal, SNR_dB)
```

**Effect on Signal:**

- Causes amplitude and phase variations
- Constellation points rotate and scale randomly
- More realistic for mobile/wireless channels
- More challenging than AWGN

---

## Training Process Flow

This is the most complex phase where the AI learns to distinguish modulation types.

### High-Level Training Flow

```
User Clicks "Train Classifier"
         ↓
┌────────────────────────────────────────┐
│  1. Generate Reference Signals         │
│     (5 modulation types)               │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  2. Extract Features (HOCs)            │
│     (10 features per modulation)       │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  3. Optimize Weights (Genetic Algo)    │
│     (50 generations)                   │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  4. Train KNN Classifier               │
│     (Store reference values)           │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  5. Return Results                     │
│     (Super-cumulants + weights)        │
└────────────────────────────────────────┘
```

### Detailed Step 1: Generate Reference Signals

```python
For each modulation_type in [BPSK, QPSK, QAM, 16-QAM, 64-QAM]:
    1. Generate clean signal (1024 samples)
    2. Apply channel effects (AWGN or Rayleigh)
    3. Store signal for feature extraction

Result: 5 reference signals
```

### Detailed Step 2: Extract HOC Features

This is where the magic happens! We extract 10 statistical features.

```
Input: Complex signal z[n], n = 0...N-1
         ↓
┌────────────────────────────────────────┐
│  2a. Compute Moments                   │
│  ────────────────────────             │
│  m_pq = E[z^(p-q) * conj(z)^q]        │
│                                         │
│  Compute 12 moments:                   │
│  • m20, m21, m22 (2nd order)           │
│  • m40, m41, m42 (4th order)           │
│  • m60, m61, m62, m63 (6th order)      │
│  • m80, m84 (8th order)                │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  2b. Calculate Cumulants               │
│  ────────────────────────             │
│  Use moment-to-cumulant formulas:      │
│                                         │
│  C1 = m20                              │
│  C2 = m21                              │
│  C3 = m40 - 3·m20·m21                  │
│  C4 = m42 - |m20|² - 2·m21²            │
│  C5 = m60 - 15·m20·m40 + 30·m20³       │
│  C6 = m61 - 5·m21·m40 - 10·m20·m41     │
│       + 30·m20²·m21                    │
│  C7 = m62 - 6·m20·m42 - 8·m21·m41      │
│       - 30·m22²·m40 + 6·m20²·m22       │
│       + 24·m21²·m22                    │
│  C8 = m63 - 9·m21·m42 + 12·m21³        │
│       - 3·m20²·m42 - 36·m22·m41        │
│       + 18·m20·m21·m22                 │
│  C9 = m80 - 35·m40² - 28·m60·m20       │
│       + 420·m40·m20² - 630·m20⁴        │
│  C10 = m84 - 16·C8·C2 + |C3|²          │
│        - 18·C4² - 72·C4·C2²            │
│        - 24·C2⁴                        │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  2c. Take Absolute Values              │
│  ────────────────────────             │
│  features = [|C1|, |C2|, ..., |C10|]   │
└────────────────────────────────────────┘
         ↓
Output: 10-dimensional feature vector
```

**Why These Features Work:**

1. **C1, C2 (2nd order)**: Capture variance and correlation
2. **C3, C4 (4th order)**: Measure kurtosis (peakedness)
3. **C5-C8 (6th order)**: Detect asymmetry and shape
4. **C9, C10 (8th order)**: Very high-order structure

**Key Properties:**

- Cumulants ≥ 3rd order are **blind to Gaussian noise**
- Different modulations have **unique cumulant signatures**
- Robust to phase and frequency offsets

### Detailed Step 3: Genetic Algorithm Optimization

Goal: Find optimal weights W = [w1, w2, ..., w10] to maximize separation.

```
Initialize Population (100 random weight vectors)
         ↓
┌────────────────────────────────────────┐
│  For Generation = 1 to 50:             │
│                                         │
│  ┌──────────────────────────────────┐ │
│  │ 3a. Evaluate Fitness              │ │
│  │ ────────────────────             │ │
│  │ For each individual:              │ │
│  │   For each mod pair (i,j):        │ │
│  │     SC_i = W · features_i         │ │
│  │     SC_j = W · features_j         │ │
│  │     distance += |SC_i - SC_j|     │ │
│  │   fitness = average_distance      │ │
│  └──────────────────────────────────┘ │
│         ↓                              │
│  ┌──────────────────────────────────┐ │
│  │ 3b. Selection                     │ │
│  │ ────────────────────             │ │
│  │ • Sort by fitness (best first)    │ │
│  │ • Keep top 25% as parents         │ │
│  │ • Discard bottom 75%              │ │
│  └──────────────────────────────────┘ │
│         ↓                              │
│  ┌──────────────────────────────────┐ │
│  │ 3c. Crossover                     │ │
│  │ ────────────────────             │ │
│  │ For i = 1 to 75:                  │ │
│  │   parent1, parent2 = random_pick  │ │
│  │   crossover_point = random(1-9)   │ │
│  │   child = [parent1[0:k],          │ │
│  │            parent2[k:10]]          │ │
│  │   offspring.append(child)         │ │
│  └──────────────────────────────────┘ │
│         ↓                              │
│  ┌──────────────────────────────────┐ │
│  │ 3d. Mutation                      │ │
│  │ ────────────────────             │ │
│  │ For each child:                   │ │
│  │   if random() < 0.1:              │ │
│  │     gene_index = random(0-9)      │ │
│  │     child[gene_index] = random()  │ │
│  └──────────────────────────────────┘ │
│         ↓                              │
│  ┌──────────────────────────────────┐ │
│  │ 3e. Create New Population         │ │
│  │ ────────────────────             │ │
│  │ population = parents + offspring  │ │
│  └──────────────────────────────────┘ │
│                                         │
└────────────────────────────────────────┘
         ↓
Return Best Individual (highest fitness)
```

**Evolution Example:**

```
Generation 1:
  Best fitness = 0.523
  Best weights = [0.12, 0.89, 0.34, 0.56, 0.23, 0.78, 0.45, 0.67, 0.91, 0.15]

Generation 10:
  Best fitness = 0.678
  Best weights = [0.08, 0.92, 0.41, 0.55, 0.19, 0.83, 0.48, 0.71, 0.94, 0.11]

Generation 25:
  Best fitness = 0.812
  Best weights = [0.05, 0.95, 0.45, 0.52, 0.15, 0.88, 0.51, 0.75, 0.97, 0.08]

Generation 50:
  Best fitness = 0.934 ← OPTIMAL!
  Best weights = [0.03, 0.98, 0.47, 0.50, 0.12, 0.91, 0.53, 0.78, 0.99, 0.05]
```

**Interpretation:**

- w2, w9 are large → C2 and C9 are most discriminative
- w1, w10 are small → C1 and C10 contribute less

### Detailed Step 4: Train KNN Classifier

```python
# Compute super-cumulant for each modulation type
super_cumulants = {}

For each modulation_type:
    features = extract_hocs(signal[modulation_type])
    SC = sum(optimized_weights[i] * features[i] for i in 0..9)
    super_cumulants[modulation_type] = SC

# Store for classification
classifier.training_data = super_cumulants
classifier.optimized_weights = optimized_weights

# Example output:
# {
#   "BPSK": 0.234,
#   "QPSK": 0.567,
#   "QAM": 0.789,
#   "16-QAM": 0.891,
#   "64-QAM": 0.923
# }
```

---

## Classification Process Flow

Once trained, classification is fast and straightforward.

```
User Clicks "Classify Signal"
         ↓
┌────────────────────────────────────────┐
│  1. Validate Prerequisites             │
│  ────────────────────────             │
│  • Check if signal is generated        │
│  • Check if classifier is trained      │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  2. Extract Features from Test Signal  │
│  ────────────────────────             │
│  • Compute 12 moments                  │
│  • Calculate 10 cumulants (C1-C10)     │
│  • Take absolute values                │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  3. Compute Test Super-Cumulant        │
│  ────────────────────────             │
│  SC_test = W* · test_features          │
│  where W* = optimized weights          │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  4. Calculate Distances                │
│  ────────────────────────             │
│  For each reference_modulation:        │
│    distance = |SC_test - SC_ref|       │
│                                         │
│  Example:                              │
│  • d_BPSK = |0.654 - 0.234| = 0.420    │
│  • d_QPSK = |0.654 - 0.567| = 0.087 ←  │
│  • d_QAM  = |0.654 - 0.789| = 0.135    │
│  • d_16QAM= |0.654 - 0.891| = 0.237    │
│  • d_64QAM= |0.654 - 0.923| = 0.269    │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  5. Find Nearest Neighbor              │
│  ────────────────────────             │
│  prediction = argmin(distances)        │
│  = QPSK (smallest distance = 0.087)    │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  6. Calculate Confidence               │
│  ────────────────────────             │
│  min_dist = 0.087                      │
│  total_dist = 0.420+0.087+0.135+       │
│               0.237+0.269 = 1.148      │
│  confidence = 1 - (0.087/1.148)        │
│             = 1 - 0.076 = 0.924 = 92.4%│
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  7. Return Results                     │
│  ────────────────────────             │
│  {                                      │
│    "predicted_modulation": "QPSK",     │
│    "confidence": 0.924,                │
│    "distances": {...},                 │
│    "features": [...]                   │
│  }                                      │
└────────────────────────────────────────┘
```

### Confidence Interpretation

```
Confidence > 0.9  → Very confident, clear signal
Confidence 0.7-0.9 → Confident, good SNR
Confidence 0.5-0.7 → Moderate, some noise
Confidence < 0.5  → Low, ambiguous or very noisy
```

---

## Algorithm Complexity Analysis

### Time Complexity

| Operation                   | Complexity | Typical Time |
| --------------------------- | ---------- | ------------ |
| Signal Generation           | O(N)       | 0.01s        |
| Moment Computation          | O(N)       | 0.05s        |
| Cumulant Calculation        | O(1)       | 0.001s       |
| GA Fitness (per individual) | O(M²)      | 0.001s       |
| GA Total (50 gen, 100 pop)  | O(G·P·M²)  | 5-10s        |
| Classification              | O(M)       | 0.001s       |

Where:

- N = number of samples (1024)
- M = number of modulation types (5)
- G = generations (50)
- P = population size (100)

### Space Complexity

| Data Structure | Space  | Size      |
| -------------- | ------ | --------- |
| Signal array   | O(N)   | ~8KB      |
| Feature vector | O(F)   | 80 bytes  |
| GA population  | O(P·F) | ~8KB      |
| Training data  | O(M·F) | 400 bytes |

---

## Decision Tree: What Happens When?

```
User Action: Generate Signal
  ├─> Signal not generated yet?
  │   ├─> YES: Generate new signal
  │   └─> NO: Overwrite previous signal
  └─> Display visualizations

User Action: Train Classifier
  ├─> Classifier already trained?
  │   ├─> YES: Retrain (overwrite old model)
  │   └─> NO: Train from scratch
  └─> Store trained model in memory

User Action: Classify Signal
  ├─> Signal generated?
  │   ├─> NO: Show error "Generate signal first"
  │   └─> YES: Continue ↓
  ├─> Classifier trained?
  │   ├─> NO: Show error "Train classifier first"
  │   └─> YES: Continue ↓
  └─> Perform classification
```

---

## Error Handling Flow

```
Every API Call
  ↓
Try:
  Execute operation
  ↓
  Success? → Return result
  ↓
Catch Exception:
  ↓
  Is it Pydantic validation error (422)?
    YES → Parse error array
         → Format: "field: message"
         → Return formatted error
  ↓
  Is it HTTP error?
    YES → Extract error detail
         → Return detail message
  ↓
  Unknown error?
    YES → Return "Unknown error occurred"
```

---

## Performance Optimization Tips

### For Better Training:

1. **Use more samples** (2048 or 4096) for more stable features
2. **Increase GA generations** (100+) for better convergence
3. **Larger population** (200+) for exploring more solutions

### For Better Classification:

1. **Higher SNR** during training (10+ dB)
2. **Match channel types** between training and testing
3. **Use same sample count** for consistency

### For Faster Processing:

1. **Train once, classify many times** (model persists in memory)
2. **Use fewer samples** (512) for quick testing
3. **Reduce GA generations** (25) for faster training (slight accuracy loss)

---

## Summary

The AMR system follows a clear three-phase workflow:

1. **Generation Phase**: Create realistic modulated signals with channel effects
2. **Training Phase**: Learn optimal feature combination using genetic algorithm
3. **Classification Phase**: Identify unknown signals using distance-based matching

The key innovation is using **higher-order cumulants** (noise-robust features) combined with **genetic algorithm optimization** (find best feature weights) to achieve high accuracy even in noisy conditions.

**Next Steps:** See [VISUALIZATION_GUIDE.md](./VISUALIZATION_GUIDE.md) to understand the UI charts, and [CODE_EXPLANATION.md](./CODE_EXPLANATION.md) for detailed code walkthrough.
