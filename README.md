# CLIMATE IMPACT VISUALIZER

---

## Overview

The **Climate Impact Visualizer** is a data-driven web application that analyzes how climatic parameters — such as **rainfall**, **temperature**, and **pesticide usage** — affect **crop yield** across various regions and crops.

This project demonstrates the complete **ML pipeline** — from **data cleaning** and **exploratory analysis** to **model training** and **interactive visualization** using **Streamlit**.

---

## Project Structure

```
Climate-Impact-Visualiser/
│
├── app/
│   └── streamlit_app.py        # Streamlit web app
│
├── data/
│   └── cleaned_climate_data.csv  # Processed dataset
│
├── models/
│   └── trained_model.joblib    # Trained Random Forest model
│
├── notebooks/
│   ├── Data_Cleaning_01.ipynb  # Data cleaning steps
│   ├── EDA_02.ipynb            # Exploratory data analysis
│   └── Modeling_03.ipynb       # Model training and evaluation
│
├── src/
│   ├── model_training.py       # Training logic (optional modular code)
│   └── utils.py                # Helper functions
│
├── reports/
│   └── figures/ (optional)     # Generated plots for documentation
│
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```

---

## Problem Statement

Climate change has a major impact on agricultural productivity.
The goal of this project is to:

* **Analyze** the relationship between climate variables and crop yield.
* **Predict** yield for given temperature, rainfall, and pesticide conditions.
* **Visualize** trends in yield and weather patterns interactively.

---

## Dataset

**Source:** [Kaggle - FAO Climate Change & Crop Production Dataset]

| Column Name            | Description                  |
| ---------------------- | ---------------------------- |
| `Area`                 | Country/Region name          |
| `Item`                 | Crop type                    |
| `Year`                 | Year of record               |
| `Avg_rainfall`         | Average annual rainfall (mm) |
| `Avg_Temp`             | Average temperature (°C)     |
| `Pesticides(tonnes)`   | Pesticide use (tonnes)       |
| `Crop_Yield(hectares)` | Crop yield per hectare       |

---

## Features

**Data Cleaning & Preprocessing**

* Handled missing values, duplicates, and inconsistent formats
* Standardized column names and datatypes

**Exploratory Data Analysis (EDA)**

* Visualized distributions and correlations
* Heatmaps for feature relationships
* Yield trends over time

**Model Building**

* Implemented **Linear Regression** and **Random Forest Regressor**
* Achieved higher accuracy with Random Forest
* Saved trained model using `joblib`

**Interactive Streamlit Dashboard**

* Select **country**, **crop**, and **climate conditions**
* Predict yield based on rainfall, temperature, and pesticide levels
* View visualizations for trends and comparisons

---

## Tech Stack

| Component           | Technology                                               |
| ------------------- | -------------------------------------------------------- |
| **Language**        | Python                                                   |
| **Libraries**       | Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, Plotly |
| **Web Framework**   | Streamlit                                                |
| **Version Control** | Git + GitHub                                             |
| **Model Storage**   | Joblib                                                   |
| **Visualization**   | Plotly Express                                           |

---

## How to Run Locally

### Clone the Repository

```bash
git clone https://github.com/Lavanya734/Climate-Impact-Visualiser.git
cd Climate-Impact-Visualiser
```

### Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate        # For Windows
source venv/bin/activate     # For Mac/Linux
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Then open your browser at **[http://localhost:8501](http://localhost:8501)**

---

## Model Performance

| Model             | R² Score | RMSE  | Comment              |
| ----------------- | -------- | ----- | -------------------- |
| Linear Regression | 0.05     | 79329 | Poor fit             |
| Random Forest     | 0.98     | 10098 | Much better accuracy |

---

## Streamlit UI Preview

> You can include a screenshot like the one you shared.

![App Screenshot](images/app_preview.png)

---

## Team Members

* **Lavanya**
* **Kimish**
* **Himanshi**

---

## Future Enhancements

* Add more climatic variables (humidity, soil moisture, CO₂ levels)
* Integrate live climate APIs (e.g., OpenWeatherMap)
* Deploy app on Streamlit Cloud / Hugging Face Spaces
* Support for region-based recommendations

---

## Conclusion

This project showcases how **machine learning** and **interactive visualization** can help understand and predict the **impact of climate on agriculture** — a step toward sustainable farming solutions.

---

## 🛠️ Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Climate-Impact-Visualiser.git
   cd Climate-Impact-Visualiser


