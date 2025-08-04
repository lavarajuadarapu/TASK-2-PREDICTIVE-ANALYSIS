# TASK-2-PREDICTIVE-ANALYSIS
# 🏏 IPL Runs Prediction using Machine Learning

This project is part of my Data Analytics internship at **CODTECH IT SOLUTIONS**.  
The objective is to build a regression model to **predict the runs scored** by a batsman in an IPL match based on features like balls faced, boundaries, and player info.

---

## 🎯 Objective

To use Linear Regression to predict `Runs Scored` using match features such as:
- Balls Faced
- Fours
- Sixes
- Batsman
- Bowler
- Team

---

## 📁 Dataset

**File:** `ipl_prediction_data.csv`  
**Fields:**
- `Match`: Match number
- `Team`: Team name
- `Batsman`: Player batting
- `Bowler`: Bowler
- `Runs Scored`: Target variable
- `Balls Faced`, `Fours`, `Sixes`: Features

---

## 🧰 Tools & Technologies

- Python 3
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

## 🧠 Machine Learning Flow

1. Load dataset
2. One-hot encode categorical features
3. Split data (train/test)
4. Train Linear Regression model
5. Predict and evaluate
6. Plot actual vs predicted runs
7. Save evaluation to `.txt` and chart to `.png`

---

## 📈 Output Example

**File:** `ipl_output.txt`

