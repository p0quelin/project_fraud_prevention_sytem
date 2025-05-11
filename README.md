# Project Fraud Prevention System

This project was created during the Data Science course by General Assembly that I followed Q2 & Q3 2021.

The goal of this project aims to develop a payment fraud detection system, able to detect fraudulent transactions following multiple fraud patterns using Python and multiple libraries (Pandas, NumPy, Seaborn, Scikit-learn, etc).

## Setup and Data Generation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Generate the dataset:
Run the `generator.ipynb` notebook or use the Python script:
```bash
python generator.py
```

This will create the required CSV files in the `data` directory. The generator is based on work by [Yann-Aël Le Borgne & Gianluca Bontempi](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html).

## Visualizations

The project includes both interactive and static visualizations for fraud analysis:

### Fraud Analysis Dashboard
![Fraud Analysis Dashboard](visualizations/fraud_analysis_dashboard.png)

The dashboard shows:
1. Distribution of transaction amounts for fraudulent vs legitimate transactions
2. Hourly fraud rate patterns
3. Terminal risk heatmap showing geographical distribution of fraud
4. Customer transaction patterns and risk analysis

### Model Performance Tracking
![Model Performance Tracking](visualizations/model_performance_tracking.png)

This visualization shows:
1. Model performance metrics over time (Precision, Recall, F1-Score)
2. Transaction volume trends
3. Fraud rate evolution

Interactive versions of these visualizations (HTML) are available in the `visualizations` directory for local use.

