# 🎓 Student Career Success Prediction

A machine learning project that predicts whether a university student will be **placed in a job**, based on academic performance, technical skills, experience, and online presence.

---

## 📌 Project Overview

This project uses a dataset of **50,000 student records** to build a binary classification model that predicts `Placement_Status` (**Placed** / **Not Placed**).

It covers the full workflow: data exploration, **data leakage detection**, preprocessing, feature engineering, **handling class imbalance with SMOTETomek**, and comparing 10+ models with hyperparameter tuning.

---

## 📂 Dataset

| Property        | Value                                |
| --------------- | ------------------------------------ |
| Source          | `student_career_success_dataset.csv` |
| Total Records   | 50,000 rows                          |
| Total Columns   | 29 columns                           |
| Target Variable | `Placement_Status`                   |
| Classes         | Placed / Not Placed                  |
| Duplicates      | 0                                    |
| Missing Values  | 0                                    |

### Columns

| Column                                                   | Type   | Description                                 |
| -------------------------------------------------------- | ------ | ------------------------------------------- |
| `Age`                                                    | int    | Student age (18 – 30)                       |
| `Gender`                                                 | object | Male, Female, Other                         |
| `University_Year`                                        | object | Freshman, Sophomore, Junior, Senior         |
| `Major`                                                  | object | CS, IT, Software Eng., AI, Data Science...  |
| `Attendance_Percentage`                                  | int    | Class attendance (50 – 100%)                |
| `Study_Hours_Per_Week`                                   | int    | Weekly study hours (5 – 45)                 |
| `CGPA`                                                   | float  | Cumulative GPA (2.0 – 4.0)                  |
| `Academic_Performance`                                   | object | Poor, Average, Good, Excellent              |
| `Programming_Skill`                                      | int    | Skill rating out of 10                      |
| `Projects_Completed`                                     | int    | Number of projects                          |
| `Certifications`                                         | int    | Number of certifications                    |
| `Hackathons`                                             | int    | Hackathons attended                         |
| `Internships`                                            | int    | Number of internships                       |
| `GitHub_Profile` / `LinkedIn_Profile`                    | object | Yes / No                                    |
| `Leadership_Experience`                                  | object | Yes / No                                    |
| `Resume_Score` / `Interview_Score`                       | int    | Resume and interview scores                 |
| `Communication_Skills` / `Teamwork` / `Problem_Solving`  | int    | Soft skill ratings out of 10                |
| `English_Proficiency`                                    | object | Basic, Intermediate, Advanced               |
| `Placement_Status`                                       | object | **Target variable**                         |

### ⚠️ Removed Columns (Data Leakage)

These columns are only known **after** a student is placed, so using them would leak the answer to the model:

| Column                | Reason                          |
| --------------------- | ------------------------------- |
| `Company_Tier`        | Exists only for placed students |
| `Career_Field`        | Exists only for placed students |
| `Placement_Mode`      | Exists only for placed students |
| `Starting_Salary_USD` | Result of placement             |
| `Employability_Score` | Derived from the outcome        |
| `Student_ID`          | Identifier, no predictive value |

---

## 🔧 Project Pipeline

```
Load Data → EDA → Remove Leakage → Split → Encode → Feature Engineering → Scale → SMOTETomek → Modeling → Hyperparameter Tuning
```

---

## 🧹 Data Cleaning & Preprocessing

- **Stratified 80/20 split** (`random_state=42`) → 40,000 train / 10,000 test
- **Gender**: rare `Other` category merged into the most frequent value, then label encoded
- **Binary columns** (`GitHub_Profile`, `LinkedIn_Profile`, `Leadership_Experience`) → Label Encoding
- **Ordinal columns** mapped in a meaningful order:

| Column                 | Mapping                                         |
| ---------------------- | ----------------------------------------------- |
| `University_Year`      | Freshman 0 < Sophomore 1 < Junior 2 < Senior 3  |
| `Academic_Performance` | Poor 0 < Average 1 < Good 2 < Excellent 3       |
| `English_Proficiency`  | Basic 0 < Intermediate 1 < Advanced 2           |

- **Major** → One-Hot Encoding with `drop='first'`
- All encoders and the scaler are **fitted on the training set only**, then applied to the test set to avoid leakage

---

## 📊 Exploratory Data Analysis (EDA)

| Chart                                 | Purpose                                     |
| ------------------------------------- | ------------------------------------------- |
| Histograms + KDE                      | Distribution of Age, Attendance, Study Hours, CGPA, Resume Score |
| Boxplots                              | Detect outliers (noticeable in `CGPA`)      |
| Count plots                           | Gender and University Year balance          |
| Bar plot: Interview Score vs Placement | Relationship between interview score and outcome |
| Correlation Heatmap                   | Find highly correlated features             |

---

## ⚙️ Feature Engineering

```python
X['Application_Score'] = (X['Resume_Score'] + X['Interview_Score']) / 2

X['Skill_Average'] = (X['Programming_Skill'] + X['Problem_Solving']
                      + X['Communication_Skills'] + X['Teamwork']) / 4
```

- Dropped the original columns that were merged or highly correlated: `Resume_Score`, `Interview_Score`, `Programming_Skill`, `Communication_Skills`, `Teamwork`, `Problem_Solving`, `Academic_Performance`, `Age`, `Gender`, `University_Year`
- Final feature count: **20 columns**

---

## ⚖️ Handling Class Imbalance

The target is imbalanced (about **78% Placed** vs **22% Not Placed**), so **SMOTETomek** was applied on the **training set only**:

| Stage            | Placed | Not Placed |
| ---------------- | ------ | ---------- |
| Before SMOTETomek | 31,232 | 8,768      |
| After SMOTETomek  | 30,963 | 30,963     |

---

## 🤖 Models

| # | Model                                   | Notes                                   |
| - | --------------------------------------- | --------------------------------------- |
| 1 | K-Nearest Neighbors                     | k = 9 and k = 33                        |
| 2 | Logistic Regression                     | Default and tuned with `GridSearchCV`   |
| 3 | Decision Tree                           | Gini, `max_depth=9`                     |
| 4 | Random Forest                           | 110 trees, and a regularized version    |
| 5 | AdaBoost                                | Default parameters                      |
| 6 | Gradient Boosting                       | Default parameters                      |
| 7 | XGBoost                                 | Manual settings and `RandomizedSearchCV` |

### Full Comparison

| Model                              | Train Acc | Test Acc   | Recall (Not Placed) |
| ---------------------------------- | --------- | ---------- | ------------------- |
| KNN (k=9)                          | 75.6%     | 67.7%      | 0.71                |
| KNN (k=33)                         | 71.7%     | 68.8%      | 0.75                |
| Logistic Regression                | 75.1%     | 74.7%      | 0.71                |
| Logistic Regression (GridSearchCV) | 75.1%     | 74.7%      | 0.71                |
| AdaBoost                           | 76.5%     | 75.6%      | 0.65                |
| Random Forest (regularized)        | 78.9%     | 77.2%      | 0.64                |
| Decision Tree (depth 9)            | 78.8%     | 77.7%      | 0.55                |
| XGBoost (manual)                   | –         | 78.0%      | 0.59                |
| Random Forest (110 trees)          | 99.6%     | 80.2%      | 0.46                |
| Gradient Boosting                  | 80.9%     | 80.4%      | 0.49                |
| **XGBoost (RandomizedSearchCV)** ✅ | 87.6%     | **81.0%**  | 0.38                |

---

## 🔍 Hyperparameter Tuning

`RandomizedSearchCV` was used for XGBoost (20 combinations × 3-fold CV = **60 fits**, scored on **F1**):

```python
param_dist_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}
```

### Best Parameters Found

```
n_estimators     : 300
max_depth        : 9
learning_rate    : 0.05
subsample        : 0.8
colsample_bytree : 0.7
```

`GridSearchCV` was used for Logistic Regression: best params `C=1`, `penalty='l1'`, `solver='liblinear'`.

### Best Model Results (XGBoost)

| Class      | Precision | Recall | F1-Score |
| ---------- | --------- | ------ | -------- |
| Not Placed | 0.60      | 0.38   | 0.47     |
| Placed     | 0.84      | 0.93   | 0.88     |
| **Accuracy** |         |        | **0.81** |

---

## 💡 Key Takeaways

- **Tree-based ensembles** (XGBoost, Gradient Boosting, Random Forest) clearly beat distance-based and linear models: about **80%** accuracy versus **68 – 75%**.
- **Accuracy is not the whole story.** Because the data is imbalanced, the most accurate models catch fewer of the students who are actually *Not Placed*. Logistic Regression and KNN have much higher recall on that class (**0.71 – 0.75**), so the best model depends on the goal: to flag at-risk students, recall on `Not Placed` matters more than overall accuracy.
- **Overfitting:** the default Random Forest reached **99.6%** train accuracy versus **80.2%** test accuracy. Limiting `max_depth` and `min_samples_leaf` reduced the gap.
- **Removing leakage columns early** is essential. Otherwise the model would look far better than it really is.

---

## 🛠️ Tech Stack

| Library                  | Usage                                  |
| ------------------------ | -------------------------------------- |
| `pandas`                 | Data loading and manipulation          |
| `numpy`                  | Numerical operations                   |
| `matplotlib` / `seaborn` | Data visualization                     |
| `scikit-learn`           | Preprocessing, models, tuning, metrics |
| `xgboost`                | Gradient boosting model                |
| `imbalanced-learn`       | SMOTETomek for class imbalance         |

---

## 🚀 How to Run

1. Clone the repo

```bash
git clone https://github.com/hebaSayed8/student-career-success-prediction-ML.git
cd student-career-success-prediction-ML
```

2. Install requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn jupyter
```

3. Update the dataset path in the notebook. It was written in Google Colab, so change:

```python
data = pd.read_csv('/content/student_career_success_dataset.csv')
```

to:

```python
data = pd.read_csv('student_career_success_dataset.csv')
```

4. Open the notebook

```bash
jupyter notebook student_placement_prediction.ipynb
```

5. Run all cells in order

---

## 📁 File Structure

```
student-career-success-prediction-ML/
│
├── student_placement_prediction.ipynb                  # Main notebook
├── student_career_success_dataset.csv   # Dataset
└── README.md                            # This file
```

---

## 📈 Results Summary

|                     | Logistic Regression | Random Forest | Gradient Boosting | XGBoost (Tuned) |
| ------------------- | ------------------- | ------------- | ----------------- | --------------- |
| Train Accuracy      | 75.1%               | 99.6%         | 80.9%             | 87.6%           |
| Test Accuracy       | 74.7%               | 80.2%         | 80.4%             | **81.0%**       |
| Recall (Not Placed) | **0.71**            | 0.46          | 0.49              | 0.38            |
| F1 (Placed)         | 0.82                | 0.88          | 0.88              | **0.88**        |
| Best Overall Model  | ❌                  | ❌            | ❌                | ✅              |

---

## 🔮 Future Improvements

- Tune the decision threshold or use `class_weight` to improve recall on `Not Placed`
- Use cross-validation for all models instead of a single train/test split
- Add feature importance and SHAP analysis to explain what drives placement
- Try LightGBM or CatBoost
- Build a small Streamlit or Gradio app for live predictions

---

## 👩‍💻 Author

**Heba Sayed** — [@hebaSayed8](https://github.com/hebaSayed8)

Built as a machine learning classification project on student career and placement data.
