# Automatic Modulation Recognition (AMR) System

An AI-powered signal classification system based on the research paper "Automatic Modulation Recognition Based on the Optimized Linear Combination of Higher-Order Cumulants" by Hussain et al. (2022).

## 🎯 Project Overview

This system implements an advanced modulation recognition framework that can automatically classify five digital modulation schemes:

- **BPSK** (Binary Phase Shift Keying)
- **QPSK** (Quadrature Phase Shift Keying)
- **QAM** (Quadrature Amplitude Modulation)
- **16-QAM** (16-Quadrature Amplitude Modulation)
- **64-QAM** (64-Quadrature Amplitude Modulation)

## 🏗️ Architecture

### Backend (FastAPI + Python)

- **Modulation Generation**: Generates digital modulated signals
- **Channel Simulation**: AWGN and Rayleigh fading channels
- **Feature Extraction**: 10 Higher-Order Cumulants (HOCs)
- **Genetic Algorithm**: Optimizes feature weights
- **KNN Classifier**: Classification using K-Nearest Neighbors

### Frontend (Next.js + React + TypeScript)

- **Interactive UI**: Real-time signal generation and visualization
- **Constellation Diagrams**: Visual representation of modulated signals
- **Time Series Plots**: Signal analysis in time domain
- **Classification Dashboard**: Results and confidence metrics

## 📋 Prerequisites

### Backend Requirements

```bash
python >= 3.8
fastapi
uvicorn
numpy
scipy
pydantic
```

### Frontend Requirements

```bash
node >= 18.0.0
next >= 14.0.0
react >= 18.0.0
typescript >= 5.0.0
recharts
lucide-react
```

## 🚀 Installation

### 1. Backend Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn numpy scipy pydantic python-multipart

# Run the server
python main.py
```

The API will be available at `http://localhost:8000`

### 2. Frontend Setup

```bash
# Create Next.js project
npx create-next-app@latest amr-frontend --typescript --tailwind --app

# Navigate to project
cd amr-frontend

# Install dependencies
npm install recharts lucide-react

# Copy the page.tsx code to app/page.tsx

# Run development server
npm run dev
```

The application will be available at `http://localhost:3000`

## 📊 API Endpoints

### Generate Signal

```http
POST /generate_signal
Content-Type: application/json

{
  "modulation_type": "BPSK",
  "num_samples": 1024,
  "snr_db": 5.0,
  "channel_type": "AWGN"
}
```

### Train Classifier

```http
POST /train_classifier
Content-Type: application/json

{
  "num_samples": 1024,
  "snr_range": [0, 10],
  "channel_type": "AWGN"
}
```

### Classify Signal

```http
POST /classify
Content-Type: application/json

{
  "signal_data": [...],
  "channel_type": "AWGN"
}
```

## 🔬 Technical Details

### Higher-Order Cumulants (HOCs)

The system extracts 10 statistical features from the received signal:

- ζ₁, ζ₂: Second-order cumulants
- ζ₃, ζ₄: Fourth-order cumulants
- ζ₅, ζ₆, ζ₇, ζ₈: Sixth-order cumulants
- ζ₉, ζ₁₀: Eighth-order cumulants

### Genetic Algorithm Optimization

The GA optimizes weights γ = [γ₁, γ₂, ..., γ₁₀] to maximize inter-class distances:

**Fitness Function**: FF = argmax ||Lᵢ - Lⱼ||² where i ≠ j

**Super-Cumulant Feature**: Lᵢ = Σ(γₖ · ζₖ)

### KNN Classification

Uses K=5 neighbors with Euclidean distance metric:
D = √((Lᵢ - L*ᵢ)(Lᵢ - L*ᵢ)\*)

## 📈 Performance Metrics

### AWGN Channel (0 dB SNR, 1024 samples)

- BPSK: 100%
- QPSK: 99.9%
- QAM: 94%
- 16-QAM: 99.9%
- 64-QAM: 99.9%

### Rayleigh Fading Channel (0 dB SNR, 1024 samples)

- BPSK: 95%
- QPSK: 96.7%
- QAM: 87%
- 16-QAM: 92%
- 64-QAM: 95%

## 🎮 Usage Guide

### 1. Generate Signal

1. Select modulation type (BPSK, QPSK, QAM, 16-QAM, 64-QAM)
2. Adjust number of samples (512-4096)
3. Set SNR in dB (0-20)
4. Choose channel type (AWGN or Rayleigh)
5. Click "Generate Signal"

### 2. Train Classifier

1. Set training parameters
2. Click "Train Classifier"
3. Wait for optimization to complete
4. View super-cumulant features

### 3. Classify Signal

1. Ensure a signal is generated
2. Ensure classifier is trained
3. Click "Classify Signal"
4. View prediction and confidence

## 🔧 Configuration

### Genetic Algorithm Parameters

- Population Size: 100
- Generations: 50
- Crossover Fraction: 0.25
- Mutation Rate: 0.1

### Signal Parameters

- Carrier Frequency: Baseband
- Symbol Rate: 1 Hz
- Oversampling: 1

## 📝 Research Citation

```bibtex
@article{hussain2022automatic,
  title={Automatic Modulation Recognition Based on the Optimized Linear Combination of Higher-Order Cumulants},
  author={Hussain, Asad and Alam, Sheraz and Ghauri, Sajjad A and Ali, Mubashir and Sherazi, Husnain Raza and Akhunzada, Adnan and Bibi, Iram and Gani, Abdullah},
  journal={Sensors},
  volume={22},
  number={19},
  pages={7488},
  year={2022},
  publisher={MDPI}
}
```

## 🛠️ Future Enhancements

1. **Deep Learning Integration**: CNN/RNN-based classifiers
2. **Real-time Processing**: Stream signal classification
3. **More Modulations**: Add FSK, PSK variants
4. **Channel Models**: Rician fading, frequency-selective fading
5. **Performance Analysis**: ROC curves, confusion matrices
6. **Deployment**: Docker containerization

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is based on open-access research published under CC BY 4.0 license.

## 🙏 Acknowledgments

- Original research by Hussain et al.
- MDPI Sensors Journal
- FastAPI and Next.js communities

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ using FastAPI, React, Next.js, and AI**
