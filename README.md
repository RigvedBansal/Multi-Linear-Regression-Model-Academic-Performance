# Multivariate Academic Performance Modeling

## 📌 Overview

This project presents a comprehensive **multivariate linear regression analysis** to study how lifestyle and academic engagement factors influence student academic performance. Specifically, it models **CGPA** as a function of **sleep duration (SLP)**, **attendance percentage (ATT)**, and **screen time (SCR)**.

The goal is not only prediction accuracy, but also **interpretability** — understanding how each factor contributes to academic outcomes using statistical metrics and geometric visualization.

---

## 📊 Dataset Description

The dataset consists of **700+ student records**, each containing:

| Feature | Description                    | Units |
| ------- | ------------------------------ | ----- |
| CGPA    | Cumulative Grade Point Average | 5–10  |
| SLP     | Average daily sleep duration   | hours |
| ATT     | Attendance percentage          | %     |
| SCR     | Daily screen time              | hours |

Synthetic data was generated with controlled correlations to reflect realistic academic behavior while maintaining statistical validity.

---

## 🧹 Data Preprocessing

Key preprocessing steps include:

* Parsing numeric values and ranges
* Handling missing or malformed entries
* Removing rows with incomplete feature vectors
* Feature standardization using **StandardScaler**

This ensures numerical stability and fair coefficient comparison during regression.

---

## 🧠 Modeling Approach

* **Model**: Multivariate Linear Regression
* **Features**: Sleep (SLP), Attendance (ATT), Screen Time (SCR)
* **Target**: CGPA
* **Train/Test Split**: 80/20

### Evaluation Metrics

* **R² Score** — goodness of fit
* **RMSE** — prediction error magnitude

These metrics are computed on unseen test data to avoid overfitting bias.

---

## 📐 3D Conditional Regression Visualization

To enhance interpretability, the project includes a **3D conditional regression plane**:

* Sleep and Attendance plotted on X–Y axes
* CGPA on Z-axis
* Screen Time fixed at its mean value

Data points are visually classified as lying **above or below** the regression plane, offering geometric intuition into model predictions.

---

## 🛠️ Tech Stack

* **Python**
* **NumPy, Pandas** — data handling
* **scikit-learn** — modeling & evaluation
* **Matplotlib** — 3D visualization

---

## 📂 Project Structure

```
├── student_data_700.csv
├── regression_analysis.py
├── README.md
```

---

## 🚀 How to Run

```bash
pip install numpy pandas matplotlib scikit-learn
python regression_analysis.py
```

---

## 📈 Results & Interpretation

* Higher **attendance** and **sleep duration** show positive association with CGPA
* Increased **screen time** correlates negatively with performance
* Standardization allows direct comparison of coefficient magnitudes

The model achieves a strong R² score, indicating meaningful explanatory power while remaining interpretable.

---

## 🎓 Academic Relevance

This project demonstrates applied skills in:

* Statistical modeling
* Feature engineering
* Model evaluation
* Scientific visualization

It is suitable for coursework, ML club submissions, internships, and research-oriented profiles.

---

## 📬 Author

Rigved Kamlesh Bansal

(BITS Pilani)
