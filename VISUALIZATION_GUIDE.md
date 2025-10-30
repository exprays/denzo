# 📊 Visualization Guide

This document explains all the charts and visualizations in the AMR System UI, their purpose, and how to interpret them.

---

## Table of Contents

1. [Overview](#overview)
2. [Constellation Diagram](#constellation-diagram)
3. [Signal Time Series](#signal-time-series)
4. [Classification Distances Chart](#classification-distances-chart)
5. [Super-Cumulant Features Chart](#super-cumulant-features-chart)
6. [Reading the Charts Together](#reading-the-charts-together)

---

## Overview

The AMR system provides four main visualizations:

```
┌─────────────────────────────────────────────┐
│  After Signal Generation:                   │
│  • Constellation Diagram                    │
│  • Signal Time Series                       │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  After Training:                            │
│  • Super-Cumulant Features Chart            │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  After Classification:                      │
│  • Classification Distances Chart           │
└─────────────────────────────────────────────┘
```

---

## Constellation Diagram

### What Is It?

A **constellation diagram** is a 2D scatter plot showing the complex signal in the I-Q plane:

- **X-axis (I)**: In-Phase component (real part)
- **Y-axis (Q)**: Quadrature component (imaginary part)
- **Each dot**: One signal sample

### Why Is It Important?

The constellation diagram is **the signature** of a modulation scheme. Each modulation type has a unique geometric pattern.

### What You Should See

#### BPSK (Binary Phase Shift Keying)

```
       Q
       ↑
       |
●──────┼──────●  Two points on I-axis
       |       (180° apart)
───────┼───────→ I
       |
```

**Expected:**

- 2 distinct clusters
- Aligned horizontally (on real axis)
- Points at approximately -1 and +1

**With Noise:**

- Clusters spread around -1 and +1
- Higher SNR → tighter clusters
- Lower SNR → overlapping clouds

#### QPSK (Quadrature Phase Shift Keying)

```
       Q
       ↑
    ●  |  ●    Four points at 45° angles
───────┼───────→ I
    ●  |  ●
       |
```

**Expected:**

- 4 distinct clusters
- Located at 45°, 135°, -45°, -135°
- Equal distance from origin
- Symmetrical pattern

**With Noise:**

- Clouds at 4 corners
- May overlap at low SNR
- Rotational symmetry preserved

#### 16-QAM (16-Quadrature Amplitude Modulation)

```
       Q
       ↑
   ●●●●        16 points in 4×4 grid
   ●●●●
───────┼───────→ I
   ●●●●
   ●●●●
       |
```

**Expected:**

- 16 distinct clusters in grid pattern
- 4 rows × 4 columns
- Unequal distances from origin
- Corner points furthest out

**With Noise:**

- Grid structure visible
- Inner points may overlap
- Outer points more separated

#### 64-QAM

```
       Q
       ↑
  ●●●●●●●●    64 points in 8×8 grid
  ●●●●●●●●
  ●●●●●●●●
  ●●●●●●●●
───────┼───────→ I
  ●●●●●●●●
  ●●●●●●●●
  ●●●●●●●●
  ●●●●●●●●
       |
```

**Expected:**

- 64 closely-packed points
- 8 rows × 8 columns
- Dense grid structure
- Clear square boundary

**With Noise:**

- Grid pattern evident at high SNR
- Points may merge at low SNR
- Requires SNR > 10 dB for clarity

### How to Interpret

| Observation              | Meaning                       |
| ------------------------ | ----------------------------- |
| Clear, distinct clusters | High SNR, good signal quality |
| Scattered points         | Low SNR, noisy channel        |
| Rotated constellation    | Carrier frequency offset      |
| Scaled constellation     | Amplitude variation           |
| Asymmetric pattern       | I/Q imbalance or interference |

### Common Patterns

**Good Signal (SNR = 15 dB):**

```
      ●●
      ●●

      ●●     Clear separation
      ●●     between clusters
```

**Moderate Noise (SNR = 5 dB):**

```
    ●●●●●
    ●●●●●   Clusters spread
    ●●●●●   but still visible

    ●●●●●
    ●●●●●
```

**High Noise (SNR = 0 dB):**

```
  ●●●●●●●●●
  ●●●●●●●●●  Almost random
  ●●●●●●●●●  scatter - very
  ●●●●●●●●●  difficult to see
  ●●●●●●●●●  pattern
```

### Practical Use

1. **Verify Signal Generation**: Check if constellation matches expected pattern
2. **Assess Signal Quality**: Tighter clusters = better quality
3. **Identify Problems**: Rotations, scaling, or distortions indicate issues
4. **Choose SNR**: If clusters overlap too much, increase SNR

---

## Signal Time Series

### What Is It?

A line chart showing how the signal's real and imaginary components change over time:

- **X-axis**: Sample index (time)
- **Y-axis**: Amplitude
- **Blue line**: Real (I) component
- **Purple line**: Imaginary (Q) component

### Why Is It Important?

Shows the **temporal behavior** of the modulated signal:

- Symbol transitions
- Amplitude variations
- Noise characteristics

### What You Should See

#### Clean Signal (High SNR)

```
Amplitude
   ↑
 1 |    ──────        ──────
   |          ────────
 0 |────
   |        ────────
-1 |──────          ──────
   └─────────────────────────→ Time

   Rectangular transitions
   Clear symbol periods
```

#### Noisy Signal (Low SNR)

```
Amplitude
   ↑
 1 |  ≈≈≈≈≈≈  ≈≈≈≈≈≈≈
   |        ≈≈≈≈≈≈≈≈
 0 |≈≈≈≈≈≈
   |      ≈≈≈≈≈≈≈≈
-1 |≈≈≈≈≈≈        ≈≈≈≈≈≈≈
   └─────────────────────────→ Time

   Noisy, jagged transitions
   Symbol boundaries unclear
```

### Interpretation by Modulation Type

#### BPSK

- **Real component**: Switches between +1 and -1
- **Imaginary component**: Always ~0 (near zero)
- **Pattern**: Real line jumps between two levels

#### QPSK

- **Both components**: Switch between +0.707 and -0.707
- **Pattern**: Both lines active, similar amplitude
- **Transitions**: Both change simultaneously

#### 16-QAM / 64-QAM

- **Both components**: Multiple amplitude levels
- **Pattern**: Staircase-like transitions
- **More levels**: 16-QAM has 4 levels, 64-QAM has 8 levels

### Key Features to Observe

| Feature                  | What It Tells You                             |
| ------------------------ | --------------------------------------------- |
| **Amplitude range**      | Signal power level                            |
| **Transition sharpness** | Bandwidth and filtering                       |
| **Noise level**          | SNR quality                                   |
| **Symmetry**             | I/Q balance                                   |
| **Number of levels**     | Modulation order (higher order = more levels) |

### Practical Use

1. **Verify Modulation**: Count distinct amplitude levels
2. **Check Noise**: Smooth lines = low noise, jagged = high noise
3. **Detect Fading**: Amplitude variations indicate fading channel
4. **Debug Issues**: Unexpected patterns suggest errors

---

## Classification Distances Chart

### What Is It?

A **bar chart** showing the computed distance from the test signal to each reference modulation type:

- **X-axis**: Modulation types (BPSK, QPSK, QAM, 16-QAM, 64-QAM)
- **Y-axis**: Distance value
- **Bar height**: Distance magnitude

### Why Is It Important?

This chart shows **how the classification decision is made**:

- Shortest bar = predicted modulation
- Shows confidence by relative bar heights
- Helps debug misclassifications

### What You Should See

#### Correct Classification (High Confidence)

```
Distance
   ↑
 1 |    ████       ████       ████       ████
   |    ████       ████       ████       ████
   |    ████       ████       ████       ████
0.5|    ████       ████       ████       ████
   |    ████       ████       ████       ████
   | ██ ████       ████       ████       ████
   | ██ ████       ████       ████       ████
   └────┴──────────┴──────────┴──────────┴─────→
   QPSK  BPSK       QAM      16-QAM    64-QAM
   ↑
   Predicted (very short bar)
```

**Interpretation:**

- QPSK bar is **much shorter** than others
- Clear winner → high confidence
- Test signal strongly matches QPSK

#### Ambiguous Classification (Low Confidence)

```
Distance
   ↑
 1 |
   |    ████
   |    ████       ████
0.5|    ████       ████       ████
   |    ████       ████       ████
   | ████████    ████████    ████████   ████
   | ████████    ████████    ████████   ████
   └────┴──────────┴──────────┴──────────┴─────→
   QPSK  BPSK       QAM      16-QAM    64-QAM
   ↑
   Predicted (but bars similar height)
```

**Interpretation:**

- QPSK bar is only **slightly shorter**
- Multiple similar bars → low confidence
- Signal is noisy or ambiguous

### Distance Values Explained

**Mathematical Definition:**

```
distance(test, reference) = |SC_test - SC_reference|

where SC = super-cumulant value
```

**Typical Ranges:**

- **0.0 - 0.1**: Very close match (>90% confidence)
- **0.1 - 0.3**: Good match (70-90% confidence)
- **0.3 - 0.5**: Moderate match (50-70% confidence)
- **>0.5**: Poor match (<50% confidence)

### Reading the Chart

#### Scenario 1: Perfect Classification

```
Distances:
  BPSK:   0.654
  QPSK:   0.023  ← Minimum (predicted)
  QAM:    0.489
  16-QAM: 0.712
  64-QAM: 0.891

Analysis:
✓ QPSK distance is 28× smaller than average
✓ Confidence = 1 - (0.023 / 2.769) = 99.2%
✓ Very confident prediction
```

#### Scenario 2: Misclassification

```
Distances:
  BPSK:   0.234
  QPSK:   0.187  ← Minimum (predicted)
  QAM:    0.156  ← Should be this!
  16-QAM: 0.423
  64-QAM: 0.567

Analysis:
✗ QPSK and QAM distances very close
✗ Confidence = 1 - (0.187 / 1.567) = 88.1%
✗ Might be wrong - signal is noisy
```

### Common Patterns

**High SNR Signal:**

```
One bar very short
Other bars tall
Clear separation
```

**Low SNR Signal:**

```
All bars similar height
No clear winner
Ambiguous result
```

**Wrong Channel Type:**

```
Unexpected modulation wins
Distances don't make sense
Retrain with correct channel
```

### Practical Use

1. **Validate Results**: Check if prediction makes sense
2. **Assess Confidence**: Look at bar height differences
3. **Debug Failures**: See which modulations are confused
4. **Improve Accuracy**: If bars too similar, increase SNR or retrain

---

## Super-Cumulant Features Chart

### What Is It?

A **bar chart** showing the computed super-cumulant value for each modulation type after training:

- **X-axis**: Modulation types
- **Y-axis**: Super-cumulant value
- **Bar height**: SC magnitude

### Why Is It Important?

Shows **how well-separated** the modulation types are in the feature space:

- Large differences = easy classification
- Small differences = difficult classification
- Monotonic pattern = good feature space

### What You Should See

#### Good Training Result

```
SC Value
   ↑
 1 |                                        ████
   |                                        ████
   |                              ████      ████
0.5|                    ████      ████      ████
   |          ████      ████      ████      ████
   |   ████   ████      ████      ████      ████
   |   ████   ████      ████      ████      ████
   └────┴──────┴─────────┴─────────┴─────────┴──→
     BPSK   QPSK      QAM      16-QAM    64-QAM

   Clear increasing/decreasing trend
   Good separation between bars
```

**Interpretation:**

- Super-cumulants are **well-separated**
- Genetic algorithm found good weights
- Classification should be accurate

#### Poor Training Result

```
SC Value
   ↑
 1 |
   |    ████   ████
   |    ████   ████      ████   ████      ████
0.5|    ████   ████      ████   ████      ████
   |    ████   ████      ████   ████      ████
   |    ████   ████      ████   ████      ████
   |    ████   ████      ████   ████      ████
   └────┴──────┴─────────┴─────────┴─────────┴──→
     BPSK   QPSK      QAM      16-QAM    64-QAM

   Similar heights
   Little separation - problematic!
```

**Interpretation:**

- Super-cumulants are **too close together**
- Genetic algorithm didn't converge well
- Classification will be difficult
- **Solution**: Retrain or increase GA generations

### Super-Cumulant Values Explained

**Mathematical Definition:**

```
SC_i = w1·C1 + w2·C2 + ... + w10·C10

where:
- w = optimized weights from GA
- C = cumulant features
```

**Physical Meaning:**

- Each modulation has a unique "fingerprint"
- SC compresses 10 features into 1 discriminative value
- Larger separation = easier to distinguish

### Typical Value Ranges

```
BPSK:   0.15 - 0.35   (lowest, simplest modulation)
QPSK:   0.35 - 0.55
QAM:    0.45 - 0.65
16-QAM: 0.65 - 0.85
64-QAM: 0.80 - 0.95   (highest, most complex)
```

**Pattern:**

- Generally **increasing** with modulation complexity
- Higher-order modulations have higher SC values
- Exceptions possible depending on GA optimization

### Reading the Chart

#### Example 1: Excellent Separation

```
Values:
  BPSK:   0.234
  QPSK:   0.523
  QAM:    0.678
  16-QAM: 0.834
  64-QAM: 0.941

Analysis:
✓ Monotonically increasing
✓ Average gap = 0.177
✓ Large separation → high accuracy expected
```

#### Example 2: Poor Separation

```
Values:
  BPSK:   0.456
  QPSK:   0.489
  QAM:    0.498
  16-QAM: 0.512
  64-QAM: 0.523

Analysis:
✗ Very close together
✗ Average gap = 0.017
✗ Small separation → low accuracy expected
✗ Action: Retrain with more samples or generations
```

### Practical Use

1. **Validate Training**: Check if values are well-separated
2. **Predict Accuracy**: Larger gaps = higher accuracy
3. **Diagnose Problems**: Similar values indicate poor training
4. **Guide Retraining**: If separation is poor, adjust training parameters

---

## Reading the Charts Together

### Complete Analysis Workflow

#### Step 1: Generate Signal

```
Look at:
  1. Constellation Diagram → Verify modulation pattern
  2. Time Series → Check signal quality

Questions:
  ✓ Does constellation match expected pattern?
  ✓ Are clusters visible and separated?
  ✓ Is noise level acceptable?
```

#### Step 2: Train Classifier

```
Look at:
  1. Super-Cumulant Chart → Verify separation

Questions:
  ✓ Are values well-separated?
  ✓ Is there a clear trend?
  ✓ No overlapping values?
```

#### Step 3: Classify Signal

```
Look at:
  1. Classification Distances → Check decision
  2. Constellation (reference) → Verify it matches

Questions:
  ✓ Is shortest bar much shorter than others?
  ✓ Does prediction make sense visually?
  ✓ Is confidence score high?
```

### Example: Complete Successful Flow

**1. Generate QPSK at 10 dB SNR**

```
Constellation Diagram:
  ● ●    Four clear clusters at 45° angles

  ● ●    ✓ Matches QPSK pattern

Time Series:
  Both I and Q oscillate between +0.7 and -0.7
  ✓ Correct amplitude range
  ✓ Low noise
```

**2. Train Classifier**

```
Super-Cumulants:
  BPSK:   0.23
  QPSK:   0.56  ← Target
  QAM:    0.78
  16-QAM: 0.89
  64-QAM: 0.93

  ✓ Good separation (Δ ≈ 0.18)
  ✓ Monotonic increase
```

**3. Classify**

```
Distances:
  BPSK:   0.512
  QPSK:   0.034  ← Minimum
  QAM:    0.289
  16-QAM: 0.467
  64-QAM: 0.598

  ✓ QPSK clearly wins
  ✓ Confidence = 98.2%
  ✓ Correct prediction!
```

### Example: Troubleshooting Failed Classification

**Problem**: Classifier predicts 16-QAM but generated QPSK

**Investigation:**

**Constellation Diagram:**

```
  ● ● ●    Expected 4 clusters (QPSK)
  ● ● ●    But seeing 16 clusters?
  ● ● ●
  ● ● ●    → Signal is wrong or very noisy!
```

**Classification Distances:**

```
  BPSK:   0.345
  QPSK:   0.267
  QAM:    0.289
  16-QAM: 0.245  ← Winner
  64-QAM: 0.312

  → All bars similar height
  → Low confidence (only 25%)
  → Ambiguous signal
```

**Root Cause:**

- SNR too low (< 3 dB)
- Clusters overlapping
- Can't distinguish modulation types

**Solution:**

1. Increase SNR to 10+ dB
2. Regenerate signal
3. Reclassify

---

## Visual Debugging Guide

### Problem: Can't See Constellation Pattern

**Symptoms:**

- Random scatter of points
- No visible clusters

**Possible Causes:**

1. SNR too low (< 0 dB)
2. Wrong channel type selected
3. Sample rate mismatch

**Solutions:**

- Increase SNR to 10+ dB
- Verify channel type
- Check number of samples

### Problem: Time Series Looks Flat

**Symptoms:**

- Lines stay at zero
- No amplitude variation

**Possible Causes:**

1. Signal not generated
2. Zero signal power
3. Frontend display issue

**Solutions:**

- Click "Generate Signal"
- Check backend logs
- Refresh page

### Problem: All Classification Bars Same Height

**Symptoms:**

- No clear winner
- Very low confidence

**Possible Causes:**

1. Classifier not trained
2. Very noisy signal
3. Wrong channel type

**Solutions:**

- Train classifier first
- Increase SNR
- Match training/testing channel types

### Problem: Super-Cumulants All Similar

**Symptoms:**

- Bars nearly same height
- No separation

**Possible Causes:**

1. GA didn't converge
2. Insufficient training samples
3. Bad random initialization

**Solutions:**

- Retrain (try multiple times)
- Increase samples to 2048+
- Increase GA generations to 100

---

## Summary

### Chart Purpose Quick Reference

| Chart                        | Primary Purpose                    | When to Use             |
| ---------------------------- | ---------------------------------- | ----------------------- |
| **Constellation Diagram**    | Verify modulation pattern          | After generating signal |
| **Time Series**              | Check signal quality               | After generating signal |
| **Classification Distances** | Understand classification decision | After classification    |
| **Super-Cumulants**          | Validate training quality          | After training          |

### Key Insights from Each Chart

**Constellation Diagram:**

- Visual fingerprint of modulation
- Immediate quality assessment
- Pattern recognition by eye

**Time Series:**

- Temporal behavior
- Noise and fading effects
- Symbol transitions

**Classification Distances:**

- Decision transparency
- Confidence assessment
- Misclassification debugging

**Super-Cumulants:**

- Feature space quality
- Training effectiveness
- Expected accuracy

---

**Next Steps:** See [CODE_EXPLANATION.md](./CODE_EXPLANATION.md) for detailed code walkthrough.
