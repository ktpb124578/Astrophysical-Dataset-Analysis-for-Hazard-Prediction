# 🚀 Astrophysical Dataset Analysis  
### A Comprehensive Exploration of Orbital Characteristics and Predictive Insights  

This project explores asteroid approach data using end-to-end data science and machine learning workflows to assess potential asteroid threats. It combines advanced preprocessing, feature engineering, neural network classification, and anomaly detection with explainability through SHAP and permutation importance.

---

## 📌 Project Highlights

- ✅ Built a predictive pipeline for identifying hazardous asteroids based on orbital and physical features  
- 📊 Processed and transformed real-world NASA asteroid data with extensive feature engineering and normalization  
- 🤖 Trained a neural network with K-Fold validation and hyperparameter tuning, achieving ~83% accuracy  
- 🧠 Integrated SHAP and permutation feature importance to enhance model interpretability  
- 🛰️ Performed anomaly detection using Isolation Forest and Z-Score analysis to flag orbital outliers  

---

## 🗂️ Dataset Overview

- **Features**: Semi-Major Axis, Aphelion Distance, Miss Distance, Eccentricity, Velocity, Approach Date, and more  
- **Data Types**: Both categorical and numerical orbital features  
- **Target Variable**: `Hazardous` – binary indicator of whether an asteroid poses a risk

---

## ⚙️ Data Preprocessing

- 🧹 **Missing Values**: Imputed using forward/backward fill, interpolation, and domain-specific logic  
- 📏 **Unit Conversion**: Standardized all distances to AU/meters, and formatted dates into datetime objects  
- 🛠️ **Feature Engineering**: Created velocity at perihelion, true anomaly, orbital energy, and “Time Until Approach”  
- 📉 **Normalization**: Applied Min-Max scaling for uniform feature ranges  

---

## 📊 Exploratory Data Analysis

- Visualized distributions using histograms and boxplots to detect skewness and outliers  
- Correlation matrix showed expected relationships (e.g., Kepler’s Law between semi-major axis & mean motion)  
- Outlier detection via z-score flagged 341 high-deviation records  

---

## 🧠 Model Building

- **Model**: Feedforward Neural Network  
  - 2 hidden layers: 32 units (ReLU) + 16 units (Tanh)  
  - Output: Sigmoid layer for binary classification  
- **Training**:  
  - 10-Fold Cross-Validation  
  - GridSearchCV for tuning learning rate and hidden layer size  
  - Achieved ~83% accuracy with AUC ≈ 0.73  

---

## 📈 Evaluation Metrics

- **Confusion Matrix**:  
  - True Positives: 24  
  - True Negatives: 728  
  - False Positives: 40  
  - False Negatives: 115  

- **ROC Curve**: Shows model sensitivity vs. specificity  
- **SHAP Plot**: Visualized feature impacts, with `Miss Distance` and `Relative Velocity` being most influential  
- **Permutation Importance**: Quantified feature contributions, validating SHAP insights

---

## 🛰️ Anomaly Detection

- **Methods**:  
  - Isolation Forest (`contamination=0.05`)  
  - Z-Score thresholding (`z > 3.05`)  

- **Evaluation**: Confusion matrix and heatmap comparing both methods  
- **Insights**: Detected several meaningful outliers with extreme velocities or orbital characteristics

---

## 📦 Tech Stack

- **Languages**: Python  
- **Libraries**: pandas, NumPy, matplotlib, seaborn, scikit-learn, TensorFlow, SHAP  
- **ML Tools**: Keras, GridSearchCV, IsolationForest  
- **Visualization**: Matplotlib, Seaborn, SHAP, heatmaps  
