# 🎬 Netflix Movies & TV Shows Clustering Analysis

## 🌐 Overview
This project applies **unsupervised machine learning** to cluster the **Netflix Movies and TV Shows dataset**, uncovering natural groupings and patterns within the platform’s content library.  
By integrating **advanced clustering algorithms**, **feature engineering**, and **dimensionality reduction**, the study aims to deliver actionable insights for **content segmentation**, **personalized recommendations**, and **targeted marketing**.

---

## 🎯 Project Objectives
- 🔍 **Identify meaningful clusters** within Netflix’s catalog using unsupervised learning.  
- 🧠 **Enhance feature representation** through text-based n-grams and numerical attributes.  
- ⚙️ **Optimize clustering performance** with dimensionality reduction and evaluate results via the **Silhouette Score**.  
- 💡 **Generate business insights** to support audience targeting and content strategy.

---

## 📊 Dataset
- **Source:** Netflix Movies and TV Shows dataset  
  *(includes columns such as `title`, `description`, `release_year`, `duration`, etc.)*  
- **Features:**  
  - Initial feature space: **145 features (TF-IDF terms)**  
  - Expanded with **n-grams (~345 features)**  
  - Reduced via **PCA**  
- **Samples:** 5,451 training samples  

---

## ⚙️ Methodology

### 🧹 Data Preprocessing
- Standardized numerical features using **`StandardScaler`**.  
- Extracted **unigrams** and **bigrams** from `description` using **`TfidfVectorizer`** (max 300 features).  
- Combined textual and numerical data, including `release_year`, `duration_value`, and **t-SNE components**.

### 🧩 Feature Engineering & Selection
- Applied **`SelectKBest`** to retain the **top 150 most significant features**.  
- Integrated **temporal** and **textual** information to enhance cluster separability.

### 📉 Dimensionality Reduction
- Performed **Principal Component Analysis (PCA)** for feature compression:  
  - **218 components** retained, explaining **80.07% variance**.  
  - *(Earlier experiments: 109 components for 90% variance on 145 features, 237 for 85% on enhanced set.)*

---

## 🤖 Clustering Models

| Algorithm | Configuration | Highlights | Silhouette Score |
|------------|----------------|-------------|------------------|
| **DBSCAN** | Varied `eps` & `min_samples` | Up to 30 clusters, 96% noise | **0.6337** (on non-noise points) |
| **K-Means** | 2 clusters | Simple partitioning | 0.0518 |
| **GMM** | 2–10 components | Probabilistic clustering | 0.0518 |
| **Hierarchical (Ward Linkage)** | 240 clusters (threshold 50) | Best structure on PCA data | 0.0561 |

---

## 📈 Evaluation
- **Metric:** Silhouette score (measures cluster cohesion & separation).  
- **Findings:**  
  - Highest score: **0.1367** (84 clusters, 109 components).  
  - Dropped to **0.0561** with 240 clusters (218 components) — indicating weak global structure despite refined features.  

---

## 🧾 Results
- **Final Model:** Hierarchical Clustering on PCA-reduced data (218 components, 150 selected features).  
- **Clusters:** 240 total clusters with **Silhouette Score = 0.0561**.  
- **Sample Labels:** `[232, 117, 21, 147, 178, 213, 13, 1, 224, 109, ...]`  
- **Visualizations:**  
  - 📉 Dendrograms reveal hierarchical relationships.  
  - 🎨 Scatter plots (`PC_1` vs `PC_2`) show significant overlap but highlight regional groupings.

---

## 🧩 Installation
```bash
git clone https://github.com/faizan1343/Netflix-Movie-and-TV-Show-Analysis.git
cd Netflix-Movie-and-TV-Show-Analysis
