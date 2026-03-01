# Student Performance Analyzer

Data analysis project exploring correlations between study habits and academic performance using Python and Pandas.

---

## Overview

This project uses Python, Pandas, Seaborn, Matplotlib, and scikit-learn to analyze student performance data.

It provides:

- Data exploration and visualization
- Predictive modeling using Linear Regression
- Batch predictions from CSV files
- Interactive single-student grade predictions

---

## Features

- Load and clean the student performance dataset (`student-mat.csv`)
- Explore data with summaries, histograms, boxplots, and correlation heatmaps
- Train a Linear Regression model to predict final grades (G3)
- Batch predictions from a CSV file (`batch_predict.py`)
- Interactive predictions for single students

---

## Installation

### Requirements

- Python 3.10+
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

### Install dependencies

```bash
pip install pandas matplotlib seaborn scikit-learn
 
 ###Project structure

 Student-performance-analyzer/
│
├── Data/
│   ├── student-mat.csv
│   ├── new_students.csv
│   └── predicted_G3.csv
│
├── src/
│   ├── main.py
│   └── batch_predict.py
│
├── .gitignore
├── LICENSE
└── README.md

![alt text](image.png)
![alt text](image-1.png)