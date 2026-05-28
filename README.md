# 🎓 Enterprise Student Performance & Placement Evaluation Platform

An end-to-end Explainable AI (XAI) career assistant dashboard built with **Python, Scikit-Learn, Pandas, and Streamlit**. This system accurately assesses student employment placement metrics using tree-based ensemble models, parses incoming text resumes natively via in-memory PDF decoders, and uses interactive background simulations to build targeted, prescriptive optimization roadmaps for students.

---

## 🌟 Key Product Architecture Features

### ✅ 7-Factor Comprehensive Profile Evaluation
Captures multi-dimensional hard and soft indices including:
- Cognitive IQ
- Past Semester GPA
- Cumulative CGPA
- Prior Internships
- Extra-Curricular Activities
- Interview Communication Scores
- Portfolio Projects

---

### ✅ Non-Technical 5-Tier Likelihood Metric
Abstracts dense statistical percentages into easy-to-read evaluation tiers:

- `VERY LIKELY`
- `LIKELY`
- `MODERATE`
- `LOW`
- `VERY LOW`

---

### ✅ In-Memory PDF Resume Core Parser
Integrates `pypdf` to extract text from uploaded multi-page resumes dynamically.

Scans documents for high-demand technical keywords such as:
- Python
- SQL
- Machine Learning
- AWS
- Data Analytics

to apply real-time portfolio score multipliers.

---

### ✅ Prescriptive "What-If" Strategy Engine
If a profile falls below recruitment eligibility thresholds, the system performs a multi-variable simulation loop to calculate the exact minimum improvements needed, such as:

- `+0.4 CGPA`
- `+1 Internship`
- `+2 Projects`

to move the student into a higher placement probability bracket.

---

# 🔬 Core Engineering Challenge: Overcoming Synthetic Bias & Data Leakage

During early model experimentation, the baseline synthetic dataset produced an artificial **100% classification accuracy**.

## 📌 Diagnosis
Data inspection revealed that the synthetic generator embedded deterministic mathematical relationships between features such as:

- `Academic_Performance`
- `CGPA`
- `Prev_Sem_Result`

This caused the Random Forest model to memorize patterns rather than generalize.

---

## ✅ Solution Applied

### 1. Strategic Feature Elimination
Removed redundant and highly collinear features:
- `Academic_Performance`
- `Prev_Sem_Result`

to break deterministic dependencies.

---

### 2. Controlled Noise Injection
Injected approximately **5% random label flipping** into target variables to simulate realistic hiring uncertainty.

Examples:
- Strong candidates failing interviews
- Average candidates succeeding due to networking or communication

---

### 3. Hyperparameter Regularization
Applied architecture constraints:
- `max_depth = 7`
- `min_samples_leaf = 8`

to reduce overfitting and improve generalization.

---

## 📈 Final Production Metrics

| Metric | Value |
|---|---|
| Accuracy | ~91.5% |
| Model Type | Random Forest Ensemble |
| Generalization | High |
| Data Leakage | Mitigated |

---

# 📁 Project Structure

```text
ML-Based-Student-Performance-Placement-Analytics-System/
│
├── college_student_placement_dataset.csv
├── train_model.py
├── app.py
├── student_placement_core.pkl
├── random_forest_model.pkl
├── scaler.pkl
├── advanced_models.pkl
└── README.md
```

---

# 🚀 Installation & Setup

## 1️⃣ Clone Repository

```bash
git clone https://github.com/SrujaPisal/ML-Based-Student-Performance-Placement-Analytics-System.git

cd ML-Based-Student-Performance-Placement-Analytics-System
```

---

## 2️⃣ Install Dependencies

Ensure Python 3.9+ is installed.

```bash
pip install pandas numpy scikit-learn streamlit pypdf matplotlib seaborn
```

---

## 3️⃣ Train Models

Run the training pipeline:

```bash
python train_model.py
```

This will:
- preprocess data
- inject noise regularization
- train ensemble models
- serialize `.pkl` artifacts

---

## 4️⃣ Launch Streamlit Dashboard

```bash
streamlit run app.py
```

---

# 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python 3 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-Learn |
| Models | Random Forest, Decision Tree |
| UI Framework | Streamlit |
| Resume Parsing | pypdf |
| Visualization | Matplotlib, Seaborn |

---

# 🎯 Key Highlights

- Explainable AI-based placement evaluation
- Resume-aware prediction engine
- Dynamic recommendation system
- Interactive simulation dashboard
- Realistic noise-handled ML pipeline
- End-to-end deployable architecture

---

# 📌 Future Improvements

- Real-time institutional dashboard integration
- Deep learning resume embeddings
- LLM-powered interview preparation assistant
- Cloud deployment using AWS/GCP
- Student performance trend analytics

---

# 👩‍💻 Author

### Sruja Pisal
---

# ⭐ If you found this project useful, consider giving it a star!
