from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from scipy import stats
import uvicorn

app = FastAPI(title="AMR System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SignalRequest(BaseModel):
    modulation_type: str
    num_samples: int = 1024
    snr_db: float = 5.0
    channel_type: str = "AWGN"

class ComplexNumber(BaseModel):
    real: float
    imag: float

class ClassificationRequest(BaseModel):
    signal_data: List[ComplexNumber]
    channel_type: str = "AWGN"

class TrainingRequest(BaseModel):
    num_samples: int = 1024
    snr_range: List[float] = [0, 10]
    channel_type: str = "AWGN"

# Modulation Generation
class ModulationGenerator:
    @staticmethod
    def generate_bpsk(num_samples: int) -> np.ndarray:
        bits = np.random.randint(0, 2, num_samples)
        return 2 * bits - 1 + 0j
    
    @staticmethod
    def generate_qpsk(num_samples: int) -> np.ndarray:
        symbols = np.random.randint(0, 4, num_samples)
        constellation = np.array([1+1j, -1+1j, 1-1j, -1-1j]) / np.sqrt(2)
        return constellation[symbols]
    
    @staticmethod
    def generate_qam(num_samples: int) -> np.ndarray:
        symbols = np.random.randint(0, 4, num_samples)
        constellation = np.array([1+1j, -1+1j, 1-1j, -1-1j]) / np.sqrt(2)
        return constellation[symbols]
    
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
    
    @staticmethod
    def generate_64qam(num_samples: int) -> np.ndarray:
        symbols = np.random.randint(0, 64, num_samples)
        i_vals = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        q_vals = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        constellation = []
        for i in i_vals:
            for q in q_vals:
                constellation.append(i + 1j*q)
        constellation = np.array(constellation) / np.sqrt(42)
        return constellation[symbols]

# Channel Models
class Channel:
    @staticmethod
    def add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
        signal_power = np.mean(np.abs(signal)**2)
        snr_linear = 10**(snr_db/10)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 
                                          1j*np.random.randn(len(signal)))
        return signal + noise
    
    @staticmethod
    def rayleigh_fading(signal: np.ndarray, snr_db: float) -> np.ndarray:
        h = (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal))) / np.sqrt(2)
        faded_signal = h * signal
        return Channel.add_awgn(faded_signal, snr_db)

# Higher-Order Cumulants Extraction
class CumulantExtractor:
    @staticmethod
    def compute_moment(signal: np.ndarray, p: int, q: int) -> complex:
        return np.mean(signal**(p-q) * np.conj(signal)**q)
    
    @staticmethod
    def extract_hocs(signal: np.ndarray) -> np.ndarray:
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
        
        return np.array([
            np.abs(c1), np.abs(c2), np.abs(c3), np.abs(c4), np.abs(c5),
            np.abs(c6), np.abs(c7), np.abs(c8), np.abs(c9), np.abs(c10)
        ])

# Genetic Algorithm for Weight Optimization
class GeneticAlgorithm:
    def __init__(self, n_genes=10, pop_size=100, generations=50):
        self.n_genes = n_genes
        self.pop_size = pop_size
        self.generations = generations
    
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
    
    def optimize(self, features_dict: dict) -> np.ndarray:
        population = np.random.rand(self.pop_size, self.n_genes)
        
        for gen in range(self.generations):
            fitness_scores = np.array([self.fitness(ind, features_dict) for ind in population])
            
            # Selection
            sorted_indices = np.argsort(fitness_scores)[::-1]
            population = population[sorted_indices]
            
            # Crossover
            n_parents = self.pop_size // 4
            parents = population[:n_parents]
            offspring = []
            
            for _ in range(self.pop_size - n_parents):
                p1, p2 = parents[np.random.choice(n_parents, 2, replace=False)]
                crossover_point = np.random.randint(1, self.n_genes)
                child = np.concatenate([p1[:crossover_point], p2[crossover_point:]])
                
                # Mutation
                if np.random.rand() < 0.1:
                    mut_idx = np.random.randint(self.n_genes)
                    child[mut_idx] = np.random.rand()
                
                offspring.append(child)
            
            population = np.vstack([parents, offspring])
        
        return population[0]

# KNN Classifier
class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.training_data = {}
        self.optimized_weights = None
    
    def train(self, features_dict: dict, weights: np.ndarray):
        self.training_data = {
            mod: np.dot(weights, feats) for mod, feats in features_dict.items()
        }
        self.optimized_weights = weights
    
    def predict(self, features: np.ndarray) -> str:
        lc_test = np.dot(self.optimized_weights, features)
        
        distances = {
            mod: np.abs(lc_test - lc_train) 
            for mod, lc_train in self.training_data.items()
        }
        
        return min(distances, key=distances.get)

# Global classifier instance
classifier = KNNClassifier()

@app.get("/")
def read_root():
    return {"message": "AMR System API", "status": "active"}

@app.post("/generate_signal")
def generate_signal(request: SignalRequest):
    try:
        generator = ModulationGenerator()
        
        # Generate modulated signal
        if request.modulation_type == "BPSK":
            signal = generator.generate_bpsk(request.num_samples)
        elif request.modulation_type == "QPSK":
            signal = generator.generate_qpsk(request.num_samples)
        elif request.modulation_type == "QAM":
            signal = generator.generate_qam(request.num_samples)
        elif request.modulation_type == "16-QAM":
            signal = generator.generate_16qam(request.num_samples)
        elif request.modulation_type == "64-QAM":
            signal = generator.generate_64qam(request.num_samples)
        else:
            raise HTTPException(status_code=400, detail="Invalid modulation type")
        
        # Apply channel
        if request.channel_type == "AWGN":
            signal = Channel.add_awgn(signal, request.snr_db)
        else:
            signal = Channel.rayleigh_fading(signal, request.snr_db)
        
        # Convert to serializable format
        signal_data = [{"real": float(s.real), "imag": float(s.imag)} for s in signal[:500]]
        
        return {
            "modulation_type": request.modulation_type,
            "num_samples": request.num_samples,
            "snr_db": request.snr_db,
            "channel_type": request.channel_type,
            "signal_data": signal_data
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train_classifier")
def train_classifier(request: TrainingRequest):
    try:
        generator = ModulationGenerator()
        extractor = CumulantExtractor()
        
        modulation_types = ["BPSK", "QPSK", "QAM", "16-QAM", "64-QAM"]
        features_dict = {}
        
        # Generate and extract features for each modulation type
        for mod_type in modulation_types:
            if mod_type == "BPSK":
                signal = generator.generate_bpsk(request.num_samples)
            elif mod_type == "QPSK":
                signal = generator.generate_qpsk(request.num_samples)
            elif mod_type == "QAM":
                signal = generator.generate_qam(request.num_samples)
            elif mod_type == "16-QAM":
                signal = generator.generate_16qam(request.num_samples)
            elif mod_type == "64-QAM":
                signal = generator.generate_64qam(request.num_samples)
            
            # Apply channel
            snr = np.mean(request.snr_range)
            if request.channel_type == "AWGN":
                signal = Channel.add_awgn(signal, snr)
            else:
                signal = Channel.rayleigh_fading(signal, snr)
            
            features = extractor.extract_hocs(signal)
            features_dict[mod_type] = features
        
        # Optimize weights using GA
        ga = GeneticAlgorithm()
        optimized_weights = ga.optimize(features_dict)
        
        # Train classifier
        classifier.train(features_dict, optimized_weights)
        
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
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify")
def classify_signal(request: ClassificationRequest):
    try:
        if classifier.optimized_weights is None:
            raise HTTPException(status_code=400, detail="Classifier not trained")
        
        # Convert signal data from ComplexNumber objects to numpy complex array
        signal = np.array([complex(s.real, s.imag) for s in request.signal_data])
        
        # Extract features
        extractor = CumulantExtractor()
        features = extractor.extract_hocs(signal)
        
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
        
        return {
            "predicted_modulation": prediction,
            "confidence": float(confidence),
            "distances": distances,
            "features": features.tolist()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/modulation_types")
def get_modulation_types():
    return {
        "types": ["BPSK", "QPSK", "QAM", "16-QAM", "64-QAM"],
        "descriptions": {
            "BPSK": "Binary Phase Shift Keying",
            "QPSK": "Quadrature Phase Shift Keying",
            "QAM": "Quadrature Amplitude Modulation",
            "16-QAM": "16-Quadrature Amplitude Modulation",
            "64-QAM": "64-Quadrature Amplitude Modulation"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)