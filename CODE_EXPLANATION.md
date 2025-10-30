# 💻 Code Explanation

A comprehensive line-by-line walkthrough of the AMR system codebase.

---

## Table of Contents

1. [Backend Code Structure](#backend-code-structure)
2. [Pydantic Models](#pydantic-models)
3. [Modulation Generator](#modulation-generator)
4. [Channel Models](#channel-models)
5. [Cumulant Extractor](#cumulant-extractor)
6. [Genetic Algorithm](#genetic-algorithm)
7. [KNN Classifier](#knn-classifier)
8. [API Endpoints](#api-endpoints)
9. [Frontend Code Structure](#frontend-code-structure)
10. [Frontend Components](#frontend-components)

---

## Backend Code Structure

```python
# File: backend/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from scipy import stats
import uvicorn
```

### Import Explanations

- **FastAPI**: Modern, fast web framework for building APIs
- **HTTPException**: For returning error responses
- **CORSMiddleware**: Allows cross-origin requests from frontend
- **BaseModel**: Pydantic base class for data validation
- **numpy**: Numerical computing library
- **scipy.stats**: Statistical functions (though not heavily used here)
- **uvicorn**: ASGI server to run FastAPI

### Application Setup

```python
app = FastAPI(title="AMR System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],         # Allow all HTTP methods
    allow_headers=["*"],         # Allow all headers
)
```

**Purpose:**

- Creates FastAPI application instance
- Enables CORS to allow frontend (localhost:3000) to call backend (localhost:8000)
- In production, you should restrict `allow_origins` to specific domains

---

## Pydantic Models

### Why Pydantic?

Pydantic provides:

1. **Automatic validation**: Ensures data types are correct
2. **JSON serialization**: Converts Python objects to/from JSON
3. **API documentation**: Auto-generates OpenAPI/Swagger docs
4. **Error messages**: Clear validation error responses

### Model Definitions

```python
class SignalRequest(BaseModel):
    modulation_type: str
    num_samples: int = 1024      # Default value
    snr_db: float = 5.0          # Default value
    channel_type: str = "AWGN"   # Default value
```

**Usage:** Request body for `/generate_signal` endpoint

**Validation:**

- `modulation_type` must be a string
- `num_samples` must be an integer (defaults to 1024)
- `snr_db` must be a float (defaults to 5.0)
- `channel_type` must be a string (defaults to "AWGN")

**Example Valid Request:**

```json
{
  "modulation_type": "QPSK",
  "num_samples": 2048,
  "snr_db": 10.0,
  "channel_type": "AWGN"
}
```

```python
class ComplexNumber(BaseModel):
    real: float
    imag: float
```

**Purpose:** Represents a complex number in JSON-serializable format

**Why Needed:** Python's `complex` type cannot be directly serialized to JSON. This model allows us to send complex signals as:

```json
{ "real": 0.707, "imag": 0.707 }
```

Instead of trying to send `0.707+0.707j` (which would fail).

```python
class ClassificationRequest(BaseModel):
    signal_data: List[ComplexNumber]
    channel_type: str = "AWGN"
```

**Usage:** Request body for `/classify` endpoint

**Structure:**

- `signal_data`: Array of complex numbers (the signal to classify)
- `channel_type`: Channel model used (should match training)

```python
class TrainingRequest(BaseModel):
    num_samples: int = 1024
    snr_range: List[float] = [0, 10]
    channel_type: str = "AWGN"
```

**Usage:** Request body for `/train_classifier` endpoint

**Parameters:**

- `num_samples`: How many samples to generate for each modulation type
- `snr_range`: [min_SNR, max_SNR] - system uses the average
- `channel_type`: Which channel model to use for training

---

## Modulation Generator

```python
class ModulationGenerator:
    @staticmethod
    def generate_bpsk(num_samples: int) -> np.ndarray:
        bits = np.random.randint(0, 2, num_samples)
        return 2 * bits - 1 + 0j
```

**Line-by-line:**

1. `bits = np.random.randint(0, 2, num_samples)`
   - Generates random binary data: [0, 1, 1, 0, 1, 0, ...]
   - `randint(0, 2)` gives 0 or 1 with equal probability
2. `return 2 * bits - 1 + 0j`
   - Maps bits to constellation:
     - If bit = 0: `2*0 - 1 = -1` → -1+0j
     - If bit = 1: `2*1 - 1 = +1` → +1+0j
   - `+ 0j` converts to complex number
   - Result: [-1+0j, +1+0j, +1+0j, -1+0j, ...]

```python
    @staticmethod
    def generate_qpsk(num_samples: int) -> np.ndarray:
        symbols = np.random.randint(0, 4, num_samples)
        constellation = np.array([1+1j, -1+1j, 1-1j, -1-1j]) / np.sqrt(2)
        return constellation[symbols]
```

**Line-by-line:**

1. `symbols = np.random.randint(0, 4, num_samples)`
   - Generates random symbols: [2, 0, 3, 1, 2, ...]
   - Each symbol represents 2 bits (2^2 = 4 possibilities)
2. `constellation = np.array([1+1j, -1+1j, 1-1j, -1-1j]) / np.sqrt(2)`
   - Defines 4 constellation points at 45°, 135°, -45°, -135°
   - Division by √2 normalizes to unit average power
   - Result: [0.707+0.707j, -0.707+0.707j, 0.707-0.707j, -0.707-0.707j]
3. `return constellation[symbols]`
   - Maps each symbol to its constellation point
   - If symbols = [2, 0, 3]:
     - symbols[0]=2 → constellation[2] = 0.707-0.707j
     - symbols[1]=0 → constellation[0] = 0.707+0.707j
     - symbols[2]=3 → constellation[3] = -0.707-0.707j

```python
    @staticmethod
    def generate_16qam(num_samples: int) -> np.ndarray:
        symbols = np.random.randint(0, 16, num_samples)
        i_vals = np.array([-3, -1, 1, 3])
        q_vals = np.array([-3, -1, 1, 3])
        constellation = []
        for i in i_vals:
            for q in q_vals:
                constellation.append(i + 1j*q)
        constellation = np.array(constellation) / np.sqrt(10)
        return constellation[symbols]
```

**Line-by-line:**

1. `symbols = np.random.randint(0, 16, num_samples)`
   - Random symbols from 0-15 (4 bits per symbol)
2. `i_vals = np.array([-3, -1, 1, 3])`
   - In-phase component values (4 levels)
3. `q_vals = np.array([-3, -1, 1, 3])`
   - Quadrature component values (4 levels)
4. ```python
   for i in i_vals:
       for q in q_vals:
           constellation.append(i + 1j*q)
   ```
   - Creates all 16 combinations (4×4 grid)
   - Example points: -3-3j, -3-1j, -3+1j, -3+3j, ...
5. `constellation = np.array(constellation) / np.sqrt(10)`
   - Normalizes to unit average power
   - √10 is the normalization factor for 16-QAM grid

**Why √10?**

```
Average power = mean(|points|²)
For 16-QAM: (-3)² + (-1)² + 1² + 3² = 20
Per dimension: 20/4 = 5
Both dimensions: 5 + 5 = 10
Normalization: √10
```

---

## Channel Models

```python
class Channel:
    @staticmethod
    def add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
        signal_power = np.mean(np.abs(signal)**2)
        snr_linear = 10**(snr_db/10)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) +
                                          1j*np.random.randn(len(signal)))
        return signal + noise
```

**Line-by-line:**

1. `signal_power = np.mean(np.abs(signal)**2)`
   - Computes average signal power
   - `np.abs(signal)` gives magnitude: |signal|
   - Squaring gives power: |signal|²
   - `np.mean()` averages over all samples
2. `snr_linear = 10**(snr_db/10)`
   - Converts SNR from dB to linear scale
   - Formula: SNR_linear = 10^(SNR_dB/10)
   - Example: 10 dB → 10^(10/10) = 10^1 = 10
3. `noise_power = signal_power / snr_linear`
   - Calculates required noise power
   - SNR = Signal Power / Noise Power
   - Therefore: Noise Power = Signal Power / SNR
4. `noise = np.sqrt(noise_power/2) * (...)`
   - Creates complex Gaussian noise
   - `noise_power/2` splits power between real and imaginary parts
   - `np.random.randn()` generates standard normal distribution (mean=0, std=1)
   - Scaling by `np.sqrt(noise_power/2)` gives correct variance
5. `np.random.randn(len(signal)) + 1j*np.random.randn(len(signal))`
   - Real part: `np.random.randn(len(signal))`
   - Imaginary part: `1j*np.random.randn(len(signal))`
   - Result: Complex white Gaussian noise
6. `return signal + noise`
   - Adds noise to signal
   - Models wireless channel with additive noise

```python
    @staticmethod
    def rayleigh_fading(signal: np.ndarray, snr_db: float) -> np.ndarray:
        h = (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal))) / np.sqrt(2)
        faded_signal = h * signal
        return Channel.add_awgn(faded_signal, snr_db)
```

**Line-by-line:**

1. `h = (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal))) / np.sqrt(2)`
   - Generates Rayleigh fading coefficients
   - Complex Gaussian with zero mean
   - Division by √2 normalizes power to 1
   - Each sample has random amplitude and phase
2. `faded_signal = h * signal`
   - Applies fading to signal
   - Element-wise multiplication
   - Each symbol experiences different channel gain
3. `return Channel.add_awgn(faded_signal, snr_db)`
   - Adds AWGN noise on top of fading
   - Models realistic wireless channel

**Rayleigh Fading Physics:**

- Models multipath propagation
- Signal reaches receiver via multiple paths
- Paths combine constructively/destructively
- Results in random amplitude variations

---

## Cumulant Extractor

This is the most mathematically complex part of the code.

```python
class CumulantExtractor:
    @staticmethod
    def compute_moment(signal: np.ndarray, p: int, q: int) -> complex:
        return np.mean(signal**(p-q) * np.conj(signal)**q)
```

**Mathematical Definition:**

```
m_pq = E[z^(p-q) * z*^q]

where:
- z = complex signal
- z* = complex conjugate
- E[·] = expectation (mean)
```

**Examples:**

```python
# m_20: E[z^2]
m_20 = compute_moment(signal, 2, 0)
     = mean(signal^2 * conj(signal)^0)
     = mean(signal^2)

# m_21: E[z * z*] = E[|z|^2]
m_21 = compute_moment(signal, 2, 1)
     = mean(signal^1 * conj(signal)^1)
     = mean(|signal|^2)

# m_42: E[z^2 * z*^2] = E[|z^2|^2]
m_42 = compute_moment(signal, 4, 2)
     = mean(signal^2 * conj(signal)^2)
     = mean(|signal^2|^2)
```

```python
    @staticmethod
    def extract_hocs(signal: np.ndarray) -> np.ndarray:
        # Compute moments
        m20 = CumulantExtractor.compute_moment(signal, 2, 0)
        m21 = CumulantExtractor.compute_moment(signal, 2, 1)
        m40 = CumulantExtractor.compute_moment(signal, 4, 0)
        m41 = CumulantExtractor.compute_moment(signal, 4, 1)
        m42 = CumulantExtractor.compute_moment(signal, 4, 2)
        m60 = CumulantExtractor.compute_moment(signal, 6, 0)
        m61 = CumulantExtractor.compute_moment(signal, 6, 1)
        m62 = CumulantExtractor.compute_moment(signal, 6, 2)
        m63 = CumulantExtractor.compute_moment(signal, 6, 3)
        m80 = CumulantExtractor.compute_moment(signal, 8, 0)
        m84 = CumulantExtractor.compute_moment(signal, 8, 4)
        m22 = CumulantExtractor.compute_moment(signal, 2, 2)
```

**Moment Collection:**

- Computes 12 different moments
- Orders: 2nd (m20, m21, m22), 4th (m40, m41, m42), 6th (m60-m63), 8th (m80, m84)
- Higher orders capture more complex statistical patterns

```python
        # Calculate cumulants from moments
        c1 = m20
        c2 = m21
        c3 = m40 - 3*m20*m21
        c4 = m42 - np.abs(m20)**2 - 2*m21**2
        c5 = m60 - 15*m20*m40 + 30*m20**3
        c6 = m61 - 5*m21*m40 - 10*m20*m41 + 30*m20**2*m21
        c7 = m62 - 6*m20*m42 - 8*m21*m41 - 30*m22**2*m40 + 6*m20**2*m22 + 24*m21**2*m22
        c8 = m63 - 9*m21*m42 + 12*m21**3 - 3*m20**2*m42 - 36*m22*m41 + 18*m20*m21*m22
        c9 = m80 - 35*m40**2 - 28*m60*m20 + 420*m40*m20**2 - 630*m20**4
        c10 = m84 - 16*c8*c2 + np.abs(c3)**2 - 18*c4**2 - 72*c4*c2**2 - 24*c2**4
```

**Cumulant Formulas Explained:**

These formulas come from cumulant theory. Key insights:

1. **C1, C2 (2nd order):** Simple moments

   ```
   c1 = m20 = E[z^2]        (variance-like)
   c2 = m21 = E[|z|^2]      (power)
   ```

2. **C3, C4 (4th order):** Remove Gaussian components
   ```
   c3 = m40 - 3·m20·m21     (kurtosis correction)
   c4 = m42 - |m20|² - 2·m21²  (cross-term correction)
   ```
3. **C5-C8 (6th order):** Higher-order shape
   - More complex formulas
   - Cancel lower-order contributions
   - Capture asymmetry and fine structure
4. **C9, C10 (8th order):** Very high-order statistics
   - Most complex formulas
   - Discriminate subtle differences
   - Computationally expensive but powerful

**Why These Formulas?**

- Based on mathematical cumulant generating function
- Designed to be **orthogonal** (independent)
- **Blind to Gaussian noise** (cumulants ≥ 3)
- Each captures different statistical property

```python
        return np.array([
            np.abs(c1), np.abs(c2), np.abs(c3), np.abs(c4), np.abs(c5),
            np.abs(c6), np.abs(c7), np.abs(c8), np.abs(c9), np.abs(c10)
        ])
```

**Final Step:**

- Takes absolute value of each cumulant
- Returns 10-dimensional feature vector
- All values are positive (magnitudes)

---

## Genetic Algorithm

```python
class GeneticAlgorithm:
    def __init__(self, n_genes=10, pop_size=100, generations=50):
        self.n_genes = n_genes          # Number of weights (10 cumulants)
        self.pop_size = pop_size        # Population size
        self.generations = generations   # Number of iterations
```

**Parameters:**

- `n_genes=10`: One weight per cumulant feature
- `pop_size=100`: 100 candidate solutions (individuals)
- `generations=50`: Evolve for 50 iterations

```python
    def fitness(self, weights: np.ndarray, features_dict: dict) -> float:
        modulations = list(features_dict.keys())
        total_distance = 0
        count = 0

        for i in range(len(modulations)):
            for j in range(i+1, len(modulations)):
                lc_i = np.dot(weights, features_dict[modulations[i]])
                lc_j = np.dot(weights, features_dict[modulations[j]])
                distance = np.abs(lc_i - lc_j)
                total_distance += distance
                count += 1

        return total_distance / count if count > 0 else 0
```

**Fitness Function Breakdown:**

1. `modulations = list(features_dict.keys())`
   - Get list of modulation types: ["BPSK", "QPSK", "QAM", "16-QAM", "64-QAM"]
2. ```python
   for i in range(len(modulations)):
       for j in range(i+1, len(modulations)):
   ```
   - Nested loop over all unique pairs
   - For 5 modulations: (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
   - Total: C(5,2) = 10 pairs
3. `lc_i = np.dot(weights, features_dict[modulations[i]])`
   - Compute weighted sum (linear combination)
   - `np.dot(weights, features)` = w1·c1 + w2·c2 + ... + w10·c10
   - This is the "super-cumulant" value
4. `distance = np.abs(lc_i - lc_j)`
   - Distance between two modulation types in feature space
   - Larger distance = easier to distinguish
5. `return total_distance / count`
   - Average distance across all pairs
   - Higher fitness = better weight vector

**Goal:** Maximize average separation between all modulation pairs.

```python
    def optimize(self, features_dict: dict) -> np.ndarray:
        # Initialize population with random weights
        population = np.random.rand(self.pop_size, self.n_genes)

        for gen in range(self.generations):
            # Evaluate fitness for each individual
            fitness_scores = np.array([self.fitness(ind, features_dict)
                                      for ind in population])

            # Selection: Keep the best
            sorted_indices = np.argsort(fitness_scores)[::-1]  # Descending order
            population = population[sorted_indices]
```

**Initialization & Evaluation:**

1. `population = np.random.rand(self.pop_size, self.n_genes)`
   - Creates 100 × 10 matrix
   - Each row is one individual (weight vector)
   - Values between 0 and 1
2. `fitness_scores = np.array([self.fitness(ind, features_dict) for ind in population])`
   - Evaluates fitness for all 100 individuals
   - List comprehension over population
   - Returns array of 100 fitness values
3. `sorted_indices = np.argsort(fitness_scores)[::-1]`
   - `np.argsort()` returns indices that would sort the array
   - `[::-1]` reverses to get descending order (best first)
   - Example: if fitness = [0.5, 0.8, 0.3], sorted_indices = [1, 0, 2]
4. `population = population[sorted_indices]`
   - Reorders population by fitness
   - Best individuals move to front

```python
            # Crossover: Create offspring
            n_parents = self.pop_size // 4  # Keep top 25%
            parents = population[:n_parents]
            offspring = []

            for _ in range(self.pop_size - n_parents):
                p1, p2 = parents[np.random.choice(n_parents, 2, replace=False)]
                crossover_point = np.random.randint(1, self.n_genes)
                child = np.concatenate([p1[:crossover_point], p2[crossover_point:]])
```

**Crossover (Reproduction):**

1. `n_parents = self.pop_size // 4`
   - Keep top 25% (100//4 = 25 best individuals)
   - Discard bottom 75%
2. `parents = population[:n_parents]`
   - Select first 25 rows (already sorted by fitness)
3. `p1, p2 = parents[np.random.choice(n_parents, 2, replace=False)]`
   - Randomly pick 2 different parents
   - `replace=False` ensures they're different
4. `crossover_point = np.random.randint(1, self.n_genes)`
   - Random point between 1 and 9 (inclusive)
   - Example: crossover_point = 5
5. `child = np.concatenate([p1[:crossover_point], p2[crossover_point:]])`
   - Take first part from parent1, second part from parent2
   - Example:
     ```
     p1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
     p2 = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
     crossover_point = 5
     child = [0.1, 0.2, 0.3, 0.4, 0.5 | 0.4, 0.3, 0.2, 0.1, 0.0]
                    ← from p1 →    |    ← from p2 →
     ```

```python
                # Mutation: Random changes
                if np.random.rand() < 0.1:  # 10% probability
                    mut_idx = np.random.randint(self.n_genes)
                    child[mut_idx] = np.random.rand()

                offspring.append(child)

            # Create new population
            population = np.vstack([parents, offspring])

        return population[0]  # Return best individual
```

**Mutation & New Generation:**

1. `if np.random.rand() < 0.1:`
   - 10% chance of mutation
   - `np.random.rand()` returns value between 0 and 1
2. `mut_idx = np.random.randint(self.n_genes)`
   - Random gene index (0-9)
3. `child[mut_idx] = np.random.rand()`
   - Replace one weight with random value
   - Introduces diversity, prevents premature convergence
4. `population = np.vstack([parents, offspring])`
   - Combine parents (25) and offspring (75)
   - New population of 100 individuals
5. `return population[0]`
   - After all generations, return best individual
   - population[0] is the fittest (due to sorting)

---

## KNN Classifier

```python
class KNNClassifier:
    def __init__(self, k=5):
        self.k = k  # Number of neighbors (not actually used in this implementation)
        self.training_data = {}
        self.optimized_weights = None
```

**Note:** This is actually a **1-NN classifier** (nearest neighbor) despite the k=5 parameter. The implementation only uses the closest reference.

```python
    def train(self, features_dict: dict, weights: np.ndarray):
        self.training_data = {
            mod: np.dot(weights, feats) for mod, feats in features_dict.items()
        }
        self.optimized_weights = weights
```

**Training Process:**

1. `{mod: np.dot(weights, feats) for mod, feats in features_dict.items()}`

   - Dictionary comprehension
   - For each modulation type, compute its super-cumulant
   - Example:

     ```python
     features_dict = {
         "BPSK": [c1, c2, ..., c10],
         "QPSK": [c1, c2, ..., c10],
         ...
     }

     training_data = {
         "BPSK": w·features_BPSK = 0.234,
         "QPSK": w·features_QPSK = 0.567,
         ...
     }
     ```

2. `self.optimized_weights = weights`
   - Store weights for later use in prediction

```python
    def predict(self, features: np.ndarray) -> str:
        lc_test = np.dot(self.optimized_weights, features)

        distances = {
            mod: np.abs(lc_test - lc_train)
            for mod, lc_train in self.training_data.items()
        }

        return min(distances, key=distances.get)
```

**Prediction Process:**

1. `lc_test = np.dot(self.optimized_weights, features)`
   - Compute test signal's super-cumulant
   - Uses same weights as training
2. ```python
   distances = {
       mod: np.abs(lc_test - lc_train)
       for mod, lc_train in self.training_data.items()
   }
   ```
   - Calculate distance to each reference modulation
   - Example:

     ```python
     lc_test = 0.654
     training_data = {"BPSK": 0.234, "QPSK": 0.567, ...}

     distances = {
         "BPSK": |0.654 - 0.234| = 0.420,
         "QPSK": |0.654 - 0.567| = 0.087,
         ...
     }
     ```
3. `return min(distances, key=distances.get)`
   - Find modulation with minimum distance
   - `key=distances.get` tells `min()` to compare by value
   - Returns the key (modulation name) with smallest distance

---

## API Endpoints

### Root Endpoint

```python
@app.get("/")
def read_root():
    return {"message": "AMR System API", "status": "active"}
```

**Purpose:** Health check endpoint to verify server is running.

### Generate Signal Endpoint

```python
@app.post("/generate_signal")
def generate_signal(request: SignalRequest):
    try:
        generator = ModulationGenerator()

        # Generate modulated signal
        if request.modulation_type == "BPSK":
            signal = generator.generate_bpsk(request.num_samples)
        elif request.modulation_type == "QPSK":
            signal = generator.generate_qpsk(request.num_samples)
        # ... other modulation types
        else:
            raise HTTPException(status_code=400, detail="Invalid modulation type")
```

**Signal Generation Logic:**

1. Create `ModulationGenerator` instance
2. Use if-elif chain to select correct generation function
3. Raise 400 error if modulation type unknown

```python
        # Apply channel
        if request.channel_type == "AWGN":
            signal = Channel.add_awgn(signal, request.snr_db)
        else:
            signal = Channel.rayleigh_fading(signal, request.snr_db)
```

**Channel Application:**

- AWGN: Add Gaussian noise only
- Anything else: Apply Rayleigh fading + noise

```python
        # Convert to serializable format
        signal_data = [{"real": float(s.real), "imag": float(s.imag)}
                      for s in signal[:500]]

        return {
            "modulation_type": request.modulation_type,
            "num_samples": request.num_samples,
            "snr_db": request.snr_db,
            "channel_type": request.channel_type,
            "signal_data": signal_data
        }
```

**Response Formatting:**

1. `[{"real": float(s.real), "imag": float(s.imag)} for s in signal[:500]]`
   - List comprehension over first 500 samples
   - Converts each complex number to dict
   - `float()` ensures JSON compatibility
2. Return dict with parameters and signal data

**Why only 500 samples?**

- Reduce JSON response size
- Faster frontend rendering
- Still enough for visualization

### Train Classifier Endpoint

```python
@app.post("/train_classifier")
def train_classifier(request: TrainingRequest):
    try:
        generator = ModulationGenerator()
        extractor = CumulantExtractor()

        modulation_types = ["BPSK", "QPSK", "QAM", "16-QAM", "64-QAM"]
        features_dict = {}

        # Generate and extract features for each modulation type
        for mod_type in modulation_types:
            # Generate signal...
            # Apply channel...
            features = extractor.extract_hocs(signal)
            features_dict[mod_type] = features
```

**Training Data Collection:**

1. Loop over all 5 modulation types
2. For each:
   - Generate signal
   - Apply channel
   - Extract 10 cumulant features
3. Store in `features_dict`

```python
        # Optimize weights using GA
        ga = GeneticAlgorithm()
        optimized_weights = ga.optimize(features_dict)

        # Train classifier
        classifier.train(features_dict, optimized_weights)
```

**Optimization & Training:**

1. Create GA instance (100 population, 50 generations)
2. Run optimization to find best weights
3. Train KNN classifier with optimized weights

```python
        # Compute super-cumulants
        super_cumulants = {
            mod: float(np.dot(optimized_weights, feats))
            for mod, feats in features_dict.items()
        }

        return {
            "status": "trained",
            "super_cumulants": super_cumulants,
            "weights": optimized_weights.tolist(),
            "features": {k: v.tolist() for k, v in features_dict.items()}
        }
```

**Response:**

- `super_cumulants`: Reference values for each modulation
- `weights`: Optimized weight vector
- `features`: Raw cumulant features (for debugging)

### Classify Signal Endpoint

```python
@app.post("/classify")
def classify_signal(request: ClassificationRequest):
    try:
        if classifier.optimized_weights is None:
            raise HTTPException(status_code=400, detail="Classifier not trained")
```

**Prerequisite Check:**

- Verify classifier has been trained
- Return 400 error if not

```python
        # Convert signal data from ComplexNumber objects to numpy complex array
        signal = np.array([complex(s.real, s.imag) for s in request.signal_data])

        # Extract features
        extractor = CumulantExtractor()
        features = extractor.extract_hocs(signal)
```

**Feature Extraction:**

1. Convert list of ComplexNumber objects to numpy array
2. Extract 10 cumulant features from test signal

```python
        # Classify
        prediction = classifier.predict(features)

        # Compute confidence
        lc_test = np.dot(classifier.optimized_weights, features)
        distances = {
            mod: float(np.abs(lc_test - lc_train))
            for mod, lc_train in classifier.training_data.items()
        }

        min_dist = min(distances.values())
        total_dist = sum(distances.values())
        confidence = 1 - (min_dist / total_dist) if total_dist > 0 else 0
```

**Classification & Confidence:**

1. `prediction = classifier.predict(features)`
   - Get predicted modulation type
2. Compute distances to all reference types
3. **Confidence Calculation:**

   ```
   confidence = 1 - (min_distance / sum_all_distances)

   If min_dist is much smaller than others:
     → confidence close to 1.0 (high)

   If all distances similar:
     → confidence close to 0.0 (low)
   ```

```python
        return {
            "predicted_modulation": prediction,
            "confidence": float(confidence),
            "distances": distances,
            "features": features.tolist()
        }
```

**Response:**

- Predicted modulation type
- Confidence score (0-1)
- Distances to all references
- Extracted features

---

## Frontend Code Structure

The frontend is built with:

- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Utility-first CSS framework
- **Recharts**: Charting library

### Key React Concepts Used

1. **useState**: Manage component state
2. **useEffect**: Side effects (not heavily used here)
3. **async/await**: Asynchronous API calls
4. **TypeScript interfaces**: Type definitions
5. **Conditional rendering**: Show/hide elements based on state

---

## Frontend Components

### State Management

```typescript
const [activeTab, setActiveTab] = useState<"generate" | "train" | "classify">(
  "generate"
);
const [modulationType, setModulationType] = useState("BPSK");
const [numSamples, setNumSamples] = useState(1024);
const [snrDb, setSnrDb] = useState(5);
const [channelType, setChannelType] = useState("AWGN");
const [signalData, setSignalData] = useState<SignalPoint[]>([]);
const [loading, setLoading] = useState(false);
const [classification, setClassification] =
  useState<ClassificationResult | null>(null);
const [trainingResult, setTrainingResult] = useState<TrainingResult | null>(
  null
);
const [error, setError] = useState<string | null>(null);
```

**State Variables:**

- `activeTab`: Which tab is selected (generate/train/classify)
- `modulationType`, `numSamples`, `snrDb`, `channelType`: Signal parameters
- `signalData`: Generated signal samples
- `loading`: API call in progress
- `classification`: Classification results
- `trainingResult`: Training results
- `error`: Error message to display

### API Call Functions

```typescript
const generateSignal = async () => {
  setLoading(true);
  setError(null);
  try {
    const response = await fetch(`${API_URL}/generate_signal`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        modulation_type: modulationType,
        num_samples: numSamples,
        snr_db: snrDb,
        channel_type: channelType,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      if (Array.isArray(errorData.detail)) {
        const errorMessages = errorData.detail
          .map(
            (err: { loc: string[]; msg: string }) =>
              `${err.loc.join(".")}: ${err.msg}`
          )
          .join(", ");
        throw new Error(`Validation error: ${errorMessages}`);
      }
      throw new Error(errorData.detail || "Failed to generate signal");
    }

    const data = await response.json();
    setSignalData(data.signal_data);
  } catch (err) {
    setError(err instanceof Error ? err.message : "Unknown error");
  } finally {
    setLoading(false);
  }
};
```

**Error Handling:**

1. `if (Array.isArray(errorData.detail))`
   - Pydantic validation errors return array
   - Each error has `loc` (field path) and `msg` (message)
2. Parse and format validation errors:
   ```typescript
   errorData.detail.map((err) => `${err.loc.join(".")}: ${err.msg}`).join(", ");
   ```
   - Example: "body.signal_data.0: Input should be valid complex"
3. `finally { setLoading(false); }`
   - Always reset loading state, even if error occurred

### Chart Data Preparation

```typescript
// Prepare chart data
const timeSeriesData = signalData.map((point, idx) => ({
  index: idx,
  real: point.real,
  imag: point.imag,
  magnitude: Math.sqrt(point.real ** 2 + point.imag ** 2),
}));

const constellationData = signalData.map((point) => ({
  x: point.real,
  y: point.imag,
}));

const distanceData = classification
  ? Object.entries(classification.distances).map(([mod, dist]) => ({
      modulation: mod,
      distance: dist,
    }))
  : [];
```

**Data Transformations:**

1. **timeSeriesData**: Add index and magnitude for line chart
2. **constellationData**: Map to x-y coordinates for scatter plot
3. **distanceData**: Convert object to array of {modulation, distance}

---

## Summary

### Backend Architecture

```
FastAPI Server
├── Pydantic Models (validation)
├── Modulation Generator (signal creation)
├── Channel Models (noise/fading)
├── Cumulant Extractor (feature extraction)
├── Genetic Algorithm (optimization)
├── KNN Classifier (classification)
└── API Endpoints (REST interface)
```

### Frontend Architecture

```
Next.js App
├── State Management (React hooks)
├── API Integration (fetch calls)
├── Data Transformation (chart prep)
└── Visualization (Recharts)
```

### Data Flow

```
User Input → API Request → Backend Processing → API Response → Frontend Update → UI Render
```

---

**Congratulations!** You now understand every part of the AMR system codebase. For more information, see:

- [README.md](./README.md) - System overview and setup
- [PROCESS_FLOW.md](./PROCESS_FLOW.md) - Algorithm workflow
- [VISUALIZATION_GUIDE.md](./VISUALIZATION_GUIDE.md) - Chart explanations
