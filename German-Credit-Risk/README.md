# German Credit Risk Analysis & Prediction

## Overview

โปรเจกต์นี้เป็นการวิเคราะห์ความเสี่ยงด้านสินเชื่อ (Credit Risk) จากชุดข้อมูล **German Credit Dataset** ซึ่งประกอบด้วยข้อมูลผู้กู้ 1,000 ราย โดยมีเป้าหมายในการ **ทำนายว่าผู้กู้จะมีความเสี่ยงดี (good) หรือ ไม่ดี (bad)** เพื่อช่วยธนาคารตัดสินใจอนุมัติสินเชื่อ

## Objective

- สำรวจและทำความเข้าใจข้อมูลเชิงลึก (EDA)
- สร้าง Feature ใหม่จากข้อมูลที่มี
- เปรียบเทียบ Machine Learning Models หลายตัว
- วิเคราะห์และอธิบายผลลัพธ์ของโมเดล (Interpretability)

## Dataset

| รายละเอียด | ค่า |
|---|---|
| จำนวนข้อมูล | 1,000 แถว |
| จำนวน Features | 10 คอลัมน์ |
| Target | Risk (good / bad) |
| Class Ratio | good 70% / bad 30% |
| Missing Values | Saving accounts (183), Checking account (394) |

**Features**: Age, Sex, Job, Housing, Saving accounts, Checking account, Credit amount, Duration, Purpose

## Project Pipeline

### Phase 1: Exploratory Data Analysis (EDA)
- Distribution analysis ของทุก feature
- Correlation heatmap
- Bivariate analysis (Age vs Credit Amount, Duration vs Credit Amount)
- **Key Insight**: คนอายุน้อย (≤25) มี bad risk rate สูงถึง 57.9%, คนไม่มีข้อมูล Checking account มี bad rate 88.3%

### Phase 2: Feature Engineering
- สร้าง **Monthly Payment** (Credit Amount ÷ Duration)
- สร้าง **Age Group** (Young / Adult / Senior)
- สร้าง **Credit per Age** (ภาระหนี้ต่ออายุ)
- จัดการ Missing Values ด้วย "unknown" category
- Label Encoding + StandardScaler
- Train/Test Split 80/20 (Stratified)

### Phase 3: Modeling & Comparison
เทรนและเปรียบเทียบ **5 โมเดล** ใน 3 สถานการณ์:

| โมเดล | Baseline AUC | SMOTE AUC | Tuned AUC |
|---|---|---|---|
| Logistic Regression | 0.7707 | 0.7649 | — |
| Random Forest | **0.7790** | 0.7640 | 0.7640 |
| Gradient Boosting | 0.7446 | 0.7408 | 0.7507 |
| SVM | 0.7673 | 0.7667 | — |
| XGBoost | — | — | — |

- ใช้ **SMOTE** จัดการ class imbalance
- ใช้ **GridSearchCV** tune hyperparameters
- **Best Model**: Random Forest (Baseline) — AUC = 0.7790

### Phase 4: Evaluation & Interpretation
- **Confusion Matrix** — เปรียบเทียบทุกโมเดล
- **ROC Curve** — ทุกโมเดลในกราฟเดียว
- **Precision-Recall Curve** — สำคัญสำหรับ imbalanced data
- **Feature Importance** — ปัจจัยที่มีผลต่อ Risk มากที่สุด
- **SHAP Values** — อธิบายการตัดสินใจของโมเดลในระดับ individual prediction

## Key Findings

1. **Checking account** เป็น feature ที่สำคัญที่สุด — คนที่ไม่มีข้อมูลบัญชีเดินสะพัดเสี่ยงมาก
2. **Duration** (ระยะเวลากู้) ยิ่งนานยิ่งเสี่ยง
3. **Credit amount** ยิ่งสูงยิ่งเสี่ยง
4. **Age** คนอายุน้อยเสี่ยงกว่าคนอายุมาก
5. โมเดล Random Forest ให้ผลดีที่สุดด้วย AUC 0.779

## Project Structure

```
German-Credit-Risk/
├── notebooks/
│   └── main.ipynb          # Jupyter notebook (EDA → Modeling → Evaluation)
├── models/
│   └── best_model.pkl      # Trained model + scaler + encoders
├── src/
│   └── main.py             # Python pipeline script
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

### Required Libraries
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- xgboost
- imbalanced-learn (SMOTE)
- shap
- joblib

## Usage

**Jupyter Notebook** (recommended):
```bash
cd notebooks
jupyter notebook main.ipynb
```

**Python Script**:
```bash
python src/main.py
```

## Tech Stack

- **Language**: Python 3.9+
- **ML**: scikit-learn, XGBoost
- **Visualization**: matplotlib, seaborn, SHAP
- **Data**: pandas, numpy

## License

MIT License
