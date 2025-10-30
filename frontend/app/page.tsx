"use client";

import React, { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
} from "recharts";
import { Camera, Activity, Radio, Zap, Brain, TrendingUp } from "lucide-react";

const API_URL = "http://localhost:8000";

interface SignalPoint {
  real: number;
  imag: number;
}

interface ClassificationResult {
  predicted_modulation: string;
  confidence: number;
  distances: Record<string, number>;
  features: number[];
}

interface TrainingResult {
  status: string;
  super_cumulants: Record<string, number>;
  weights: number[];
  features: Record<string, number[]>;
}

interface TrainingProgress {
  stage: string;
  modulation?: string;
  generation?: number;
  fitness?: number;
}

const AMRSystem: React.FC = () => {
  const [activeTab, setActiveTab] = useState<"train" | "generate" | "classify">(
    "train"
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
  const [trainingProgress, setTrainingProgress] = useState<TrainingProgress[]>(
    []
  );
  const [trainingStage, setTrainingStage] = useState<string>("");

  const modulationTypes = ["BPSK", "QPSK", "QAM", "16-QAM", "64-QAM"];

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

  const trainClassifier = async () => {
    setLoading(true);
    setError(null);
    setTrainingProgress([]);
    setTrainingStage("");

    try {
      // Simulate progress updates
      const progressSteps = [
        "Initializing training...",
        "Generating BPSK training signals...",
        "Generating QPSK training signals...",
        "Generating QAM training signals...",
        "Generating 16-QAM training signals...",
        "Generating 64-QAM training signals...",
        "Extracting higher-order cumulants...",
        "Running genetic algorithm optimization...",
        "Training KNN classifier...",
        "Finalizing model...",
      ];

      let currentStep = 0;
      const progressInterval = setInterval(() => {
        if (currentStep < progressSteps.length) {
          setTrainingStage(progressSteps[currentStep]);
          setTrainingProgress((prev) => [
            ...prev,
            {
              stage: progressSteps[currentStep],
              modulation: modulationTypes[currentStep - 1] || undefined,
            },
          ]);
          currentStep++;
        }
      }, 800);

      const response = await fetch(`${API_URL}/train_classifier`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          num_samples: numSamples,
          snr_range: [0, 10],
          channel_type: channelType,
        }),
      });

      clearInterval(progressInterval);

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
        throw new Error(errorData.detail || "Failed to train classifier");
      }

      const data = await response.json();
      setTrainingResult(data);
      setTrainingStage("Training completed successfully!");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  const classifySignal = async () => {
    if (signalData.length === 0) {
      setError("Please generate a signal first");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_URL}/classify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          signal_data: signalData,
          channel_type: channelType,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        // Handle Pydantic validation errors (422) which return an array
        if (Array.isArray(errorData.detail)) {
          const errorMessages = errorData.detail
            .map(
              (err: { loc: string[]; msg: string }) =>
                `${err.loc.join(".")}: ${err.msg}`
            )
            .join(", ");
          throw new Error(`Validation error: ${errorMessages}`);
        }
        throw new Error(errorData.detail || "Failed to classify signal");
      }

      const data = await response.json();
      setClassification(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

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

  const superCumulantData = trainingResult
    ? Object.entries(trainingResult.super_cumulants).map(([mod, value]) => ({
        modulation: mod,
        value: value,
      }))
    : [];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-2xl shadow-xl p-8 mb-6">
          <div className="flex items-center gap-4 mb-4">
            <div className="bg-gradient-to-r from-blue-500 to-indigo-600 p-3 rounded-xl">
              <Radio className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-800">
                Automatic Modulation Recognition System
              </h1>
              <p className="text-gray-600 mt-1">
                AI-Powered Signal Classification using Higher-Order Cumulants &
                Genetic Algorithm
              </p>
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="bg-white rounded-2xl shadow-xl mb-6">
          <div className="flex border-b">
            <button
              onClick={() => setActiveTab("train")}
              className={`flex-1 py-4 px-6 font-semibold transition-all ${
                activeTab === "train"
                  ? "bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-tl-2xl"
                  : "text-gray-600 hover:bg-gray-50"
              }`}
            >
              <Brain className="w-5 h-5 inline mr-2" />
              Train Classifier
            </button>
            <button
              onClick={() => setActiveTab("generate")}
              className={`flex-1 py-4 px-6 font-semibold transition-all ${
                activeTab === "generate"
                  ? "bg-gradient-to-r from-blue-500 to-indigo-600 text-white"
                  : "text-gray-600 hover:bg-gray-50"
              }`}
            >
              <Activity className="w-5 h-5 inline mr-2" />
              Generate Signal
            </button>
            <button
              onClick={() => setActiveTab("classify")}
              className={`flex-1 py-4 px-6 font-semibold transition-all ${
                activeTab === "classify"
                  ? "bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-tr-2xl"
                  : "text-gray-600 hover:bg-gray-50"
              }`}
            >
              <TrendingUp className="w-5 h-5 inline mr-2" />
              Classify
            </button>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-6 rounded-lg">
            <p className="text-red-700">{error}</p>
          </div>
        )}

        {/* Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Controls */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-2xl shadow-xl p-6">
              <h2 className="text-xl font-bold text-gray-800 mb-4">
                Parameters
              </h2>

              {activeTab === "generate" && (
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Modulation Type
                    </label>
                    <select
                      value={modulationType}
                      onChange={(e) => setModulationType(e.target.value)}
                      className="w-full text-gray-600 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      {modulationTypes.map((type) => (
                        <option key={type} value={type}>
                          {type}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Number of Samples: {numSamples}
                    </label>
                    <input
                      type="range"
                      min="512"
                      max="4096"
                      step="512"
                      value={numSamples}
                      onChange={(e) => setNumSamples(parseInt(e.target.value))}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      SNR (dB): {snrDb}
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="20"
                      step="1"
                      value={snrDb}
                      onChange={(e) => setSnrDb(parseInt(e.target.value))}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Channel Type
                    </label>
                    <select
                      value={channelType}
                      onChange={(e) => setChannelType(e.target.value)}
                      className="w-full text-gray-500 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="AWGN">AWGN</option>
                      <option value="Rayleigh">Rayleigh Fading</option>
                    </select>
                  </div>

                  <button
                    onClick={generateSignal}
                    disabled={loading}
                    className="w-full bg-gradient-to-r from-blue-500 to-indigo-600 text-white py-3 rounded-lg font-semibold hover:from-blue-600 hover:to-indigo-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {loading ? "Generating..." : "Generate Signal"}
                  </button>
                </div>
              )}

              {activeTab === "train" && (
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Training Samples: {numSamples}
                    </label>
                    <input
                      type="range"
                      min="512"
                      max="4096"
                      step="512"
                      value={numSamples}
                      onChange={(e) => setNumSamples(parseInt(e.target.value))}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Channel Type
                    </label>
                    <select
                      value={channelType}
                      onChange={(e) => setChannelType(e.target.value)}
                      className="w-full text-gray-500 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="AWGN">AWGN</option>
                      <option value="Rayleigh">Rayleigh Fading</option>
                    </select>
                  </div>

                  <button
                    onClick={trainClassifier}
                    disabled={loading}
                    className="w-full bg-gradient-to-r from-green-500 to-emerald-600 text-white py-3 rounded-lg font-semibold hover:from-green-600 hover:to-emerald-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {loading ? "Training..." : "Train Classifier"}
                  </button>

                  {/* Training Progress */}
                  {loading && trainingStage && (
                    <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
                      <div className="flex items-center gap-2 mb-2">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                        <p className="text-sm font-semibold text-blue-800">
                          {trainingStage}
                        </p>
                      </div>
                      <div className="w-full bg-blue-200 rounded-full h-2">
                        <div
                          className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                          style={{
                            width: `${(trainingProgress.length / 10) * 100}%`,
                          }}
                        ></div>
                      </div>
                    </div>
                  )}

                  {trainingResult && !loading && (
                    <div className="mt-4 p-4 bg-green-50 rounded-lg border border-green-200">
                      <p className="text-green-800 font-semibold flex items-center gap-2">
                        <span className="text-xl">✓</span>
                        Classifier Trained Successfully
                      </p>
                      <p className="text-xs text-green-700 mt-1">
                        {modulationTypes.length} modulations • {numSamples}{" "}
                        samples
                      </p>
                    </div>
                  )}
                </div>
              )}

              {activeTab === "classify" && (
                <div className="space-y-4">
                  <div className="p-4 bg-blue-50 rounded-lg">
                    <p className="text-sm text-blue-800">
                      Generate a signal first, then classify it using the
                      trained model.
                    </p>
                  </div>

                  <button
                    onClick={classifySignal}
                    disabled={loading || signalData.length === 0}
                    className="w-full bg-gradient-to-r from-purple-500 to-pink-600 text-white py-3 rounded-lg font-semibold hover:from-purple-600 hover:to-pink-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {loading ? "Classifying..." : "Classify Signal"}
                  </button>

                  {classification && (
                    <div className="mt-4 p-4 bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg">
                      <h3 className="font-bold text-lg text-purple-900 mb-2">
                        Classification Result
                      </h3>
                      <p className="text-2xl font-bold text-purple-600 mb-2">
                        {classification.predicted_modulation}
                      </p>
                      <p className="text-sm text-gray-700">
                        Confidence:{" "}
                        {(classification.confidence * 100).toFixed(2)}%
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Visualizations */}
          <div className="lg:col-span-2 space-y-6">
            {/* Constellation Diagram */}
            {signalData.length > 0 && (
              <div className="bg-white rounded-2xl shadow-xl p-6">
                <h3 className="text-lg font-bold text-gray-800 mb-4">
                  Constellation Diagram
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      type="number"
                      dataKey="x"
                      name="In-Phase"
                      label={{
                        value: "In-Phase (I)",
                        position: "insideBottom",
                        offset: -5,
                      }}
                    />
                    <YAxis
                      type="number"
                      dataKey="y"
                      name="Quadrature"
                      label={{
                        value: "Quadrature (Q)",
                        angle: -90,
                        position: "insideLeft",
                      }}
                    />
                    <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                    <Scatter
                      name="Signal"
                      data={constellationData}
                      fill="#3b82f6"
                      fillOpacity={0.6}
                    />
                  </ScatterChart>
                </ResponsiveContainer>
                <p className="text-xs text-gray-600 mt-2">
                  📊 Shows the in-phase (I) and quadrature (Q) components of the
                  modulated signal. Each point represents a symbol in the
                  complex plane.
                </p>
              </div>
            )}

            {/* Time Series */}
            {signalData.length > 0 && (
              <div className="bg-white rounded-2xl shadow-xl p-6">
                <h3 className="text-lg font-bold text-gray-800 mb-4">
                  Signal Time Series
                </h3>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={timeSeriesData.slice(0, 200)}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="index"
                      label={{
                        value: "Sample Index",
                        position: "insideBottom",
                        offset: -5,
                      }}
                    />
                    <YAxis
                      label={{
                        value: "Amplitude",
                        angle: -90,
                        position: "insideLeft",
                      }}
                    />
                    <Tooltip />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="real"
                      stroke="#3b82f6"
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="imag"
                      stroke="#8b5cf6"
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
                <p className="text-xs text-gray-600 mt-2">
                  ⏱️ Time-domain representation showing real (I) and imaginary
                  (Q) signal components over the first 200 samples.
                </p>
              </div>
            )}

            {/* Classification Distances */}
            {classification && activeTab === "classify" && (
              <div className="bg-white rounded-2xl shadow-xl p-6">
                <h3 className="text-lg font-bold text-gray-800 mb-4">
                  Classification Distances
                </h3>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={distanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="modulation"
                      label={{
                        value: "Modulation Type",
                        position: "insideBottom",
                        offset: -5,
                      }}
                    />
                    <YAxis
                      label={{
                        value: "Euclidean Distance",
                        angle: -90,
                        position: "insideLeft",
                      }}
                    />
                    <Tooltip />
                    <Bar dataKey="distance" fill="#8b5cf6" />
                  </BarChart>
                </ResponsiveContainer>
                <p className="text-xs text-gray-600 mt-2">
                  🎯 Weighted Euclidean distances to each reference modulation.
                  Lower distance = better match. The shortest bar indicates the
                  predicted class.
                </p>
              </div>
            )}

            {/* Super Cumulants - Classify Tab */}
            {classification && activeTab === "classify" && (
              <div className="bg-white rounded-2xl shadow-xl p-6">
                <h3 className="text-lg font-bold text-gray-800 mb-4">
                  Super-Cumulant Feature Comparison
                </h3>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={superCumulantData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="modulation"
                      label={{
                        value: "Modulation Type",
                        position: "insideBottom",
                        offset: -5,
                      }}
                    />
                    <YAxis
                      label={{
                        value: "Super-Cumulant Value",
                        angle: -90,
                        position: "insideLeft",
                      }}
                    />
                    <Tooltip />
                    <Bar dataKey="value" fill="#10b981" />
                  </BarChart>
                </ResponsiveContainer>
                <p className="text-xs text-gray-600 mt-2">
                  🔬 Weighted sum of all 10 cumulant features for each reference
                  modulation. Shows how close the input signal is to each
                  trained class.
                </p>
              </div>
            )}

            {/* Super Cumulants - Training Tab */}
            {trainingResult && activeTab === "train" && (
              <div className="bg-white rounded-2xl shadow-xl p-6">
                <h3 className="text-lg font-bold text-gray-800 mb-4">
                  Super-Cumulant Reference Features
                </h3>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={superCumulantData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="modulation"
                      label={{
                        value: "Modulation Type",
                        position: "insideBottom",
                        offset: -5,
                      }}
                    />
                    <YAxis
                      label={{
                        value: "Super-Cumulant Value",
                        angle: -90,
                        position: "insideLeft",
                      }}
                    />
                    <Tooltip />
                    <Bar dataKey="value" fill="#10b981" />
                  </BarChart>
                </ResponsiveContainer>
                <p className="text-xs text-gray-600 mt-2">
                  🔬 Weighted sum of all 10 cumulant features for each
                  modulation type. These are the reference points stored in the
                  KNN classifier.
                </p>
              </div>
            )}

            {/* Training Progress Timeline */}
            {activeTab === "train" && trainingProgress.length > 0 && (
              <div className="bg-white rounded-2xl shadow-xl p-6">
                <h3 className="text-lg font-bold text-gray-800 mb-4">
                  Training Progress Timeline
                </h3>
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {trainingProgress.map((step, idx) => (
                    <div key={idx} className="flex items-start gap-3">
                      <div
                        className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                          idx === trainingProgress.length - 1
                            ? "bg-green-500 animate-pulse"
                            : "bg-blue-500"
                        }`}
                      >
                        <span className="text-white text-xs font-bold">
                          {idx + 1}
                        </span>
                      </div>
                      <div className="flex-1 pb-3 border-b border-gray-200">
                        <p className="text-sm font-medium text-gray-800">
                          {step.stage}
                        </p>
                        {step.modulation && (
                          <p className="text-xs text-gray-500 mt-1">
                            Processing {step.modulation}
                          </p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Optimized Weights Visualization */}
            {trainingResult && activeTab === "train" && (
              <div className="bg-white rounded-2xl shadow-xl p-6">
                <h3 className="text-lg font-bold text-gray-800 mb-4">
                  Optimized Feature Weights (GA Output)
                </h3>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart
                    data={trainingResult.weights.map((w, idx) => ({
                      feature: `C${idx + 1}`,
                      weight: w,
                    }))}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="feature"
                      label={{
                        value: "Cumulant Feature",
                        position: "insideBottom",
                        offset: -5,
                      }}
                    />
                    <YAxis
                      label={{
                        value: "Weight Value",
                        angle: -90,
                        position: "insideLeft",
                      }}
                    />
                    <Tooltip />
                    <Bar dataKey="weight" fill="#f59e0b" />
                  </BarChart>
                </ResponsiveContainer>
                <p className="text-xs text-gray-600 mt-2">
                  ⚖️ GA-optimized weights for each cumulant feature (C1-C10).
                  Higher weights indicate more discriminative features for
                  classification.
                </p>
              </div>
            )}

            {/* Features Heatmap */}
            {trainingResult && activeTab === "train" && (
              <div className="bg-white rounded-2xl shadow-xl p-6">
                <h3 className="text-lg font-bold text-gray-800 mb-4">
                  Extracted Cumulant Features (HOCs)
                </h3>
                <div className="overflow-x-auto">
                  <table className="min-w-full text-xs">
                    <thead>
                      <tr className="bg-gray-100">
                        <th className="px-3 text-gray-500 py-2 text-left font-semibold">
                          Modulation
                        </th>
                        {Array.from({ length: 10 }, (_, i) => (
                          <th
                            key={i}
                            className="px-2 py-2 text-gray-500 text-center font-semibold"
                          >
                            C{i + 1}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(trainingResult.features).map(
                        ([mod, feats]) => (
                          <tr key={mod} className="border-b hover:bg-gray-50">
                            <td className="px-3 py-2 font-medium text-gray-800">
                              {mod}
                            </td>
                            {(feats as number[]).map((feat, idx) => (
                              <td
                                key={idx}
                                className="px-2 py-2 text-center"
                                style={{
                                  backgroundColor: `rgba(59, 130, 246, ${
                                    feat / Math.max(...(feats as number[]))
                                  })`,
                                  color:
                                    feat / Math.max(...(feats as number[])) >
                                    0.5
                                      ? "white"
                                      : "black",
                                }}
                              >
                                {feat.toFixed(2)}
                              </td>
                            ))}
                          </tr>
                        )
                      )}
                    </tbody>
                  </table>
                </div>
                <p className="text-xs text-gray-600 mt-2">
                  Color intensity shows relative feature magnitude
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Info Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-6">
          <div className="bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl shadow-xl p-6 text-white">
            <Zap className="w-10 h-10 mb-3" />
            <h3 className="text-lg font-bold mb-2">Higher-Order Cumulants</h3>
            <p className="text-sm opacity-90">
              10 statistical features extracted from the received signal for
              robust classification
            </p>
          </div>
          <div className="bg-gradient-to-br from-green-500 to-emerald-600 rounded-2xl shadow-xl p-6 text-white">
            <Brain className="w-10 h-10 mb-3" />
            <h3 className="text-lg font-bold mb-2">Genetic Algorithm</h3>
            <p className="text-sm opacity-90">
              Optimizes feature weights to maximize inter-class distances
            </p>
          </div>
          <div className="bg-gradient-to-br from-purple-500 to-pink-600 rounded-2xl shadow-xl p-6 text-white">
            <TrendingUp className="w-10 h-10 mb-3" />
            <h3 className="text-lg font-bold mb-2">KNN Classifier</h3>
            <p className="text-sm opacity-90">
              K-Nearest Neighbor algorithm for accurate modulation recognition
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AMRSystem;
