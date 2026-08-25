# 🫀 CardioRisk AI: A Clinical Decision Support System

Hey! 👋 If you're checking out this repo, this is a project I built to tackle a major real-world issue in medical machine learning: **models that pretend to know things they don't.**

Most machine learning projects on the standard 70k-row Kaggle cardiovascular dataset just blindly force a "yes/no" prediction for every patient. But in real-world healthcare, forcing a diagnosis when a patient's biological markers are completely ambiguous is dangerous. This project takes a different route.

## 💡 The Core Idea: Clinical Rejection
If a traditional model tries to predict everything, it hits a hard mathematical accuracy ceiling of around 73.5% because some patient records are genuinely overlapping or unclear[cite: 1]. 

Instead of playing guessing games, this system acts like a smart medical assistant:
1. **It diagnoses clearly** when it is confident[cite: 1].
2. **It flags the gray zone** (patients with probability scores between 0.15 and 0.65) and **routes them to a human doctor** for manual review[cite: 1].

By letting go of the ambiguous cases, the model's accuracy and safety skyrocket on the patients it *does* handle autonomously[cite: 1].

---

## 🛠️ How I Built It

* **Deep Feature Engineering**: I didn't just use the 11 raw columns. I expanded them into over 100 features—adding clinical metrics like BMI, pulse pressure, and Mean Arterial Pressure (MAP), alongside polynomial interactions, Target Encoding, Principal Component Analysis (PCA), and K-Means clustering to spot patient health archetypes[cite: 1].
* **Data Cleaning**: Applied Tomek Links to clean up the training decision boundary by stripping out extremely noisy, overlapping patient records[cite: 1].
* **Stacked Meta-Ensemble**: Built a two-level stacking architecture using 5-fold stratified cross-validation[cite: 1]:
  * *Level 1*: Trained three heavy-hitting gradient boosters in parallel—**XGBoost**, **CatBoost**, and **LightGBM**[cite: 1].
  * *Level 2*: A **Logistic Regression** meta-learner that learns how to weigh each model's predictions dynamically[cite: 1].

---

## 📊 Results (On the High-Confidence Patient Subset)

When focusing on the roughly 60% of patients where the AI has high confidence, the system heavily prioritizes **Recall** to ensure we don't miss at-risk patients[cite: 1]:

* **Accuracy**: 81.68%[cite: 1]
* **Recall (Sensitivity)**: 94.07%[cite: 1]
* **Precision**: 79.81%[cite: 1]
* **F1-Score**: 0.8635[cite: 1]
* **ROC-AUC**: 0.8267[cite: 1]

*(Check out the `plots/` folder to see the radar charts and ROC curves visualizing these results!)*

---

## 🚀 Tech Stack
* **Language**: Python
* **ML & Stats**: Scikit-Learn, XGBoost, CatBoost, LightGBM, Imbalanced-Learn, Category Encoders
* **Data & Viz**: Pandas, NumPy, Matplotlib

Feel free to look around the code, and let me know if you want to chat about the architecture!
