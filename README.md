# Risk Alert Classifier

A Streamlit project that builds, evaluates and optimizes classification models to predict high-risk customer behaviour (`risk_status`: 1 = risky, 0 = safe).

## Video

**Video link:** https://drive.google.com/file/d/1gC931ojG6hdRIomPp1Fv3qwmKrEWJACD/view?usp=sharing

## Project Objective

Design, evaluate and optimize a classification system that identifies risky customers, with a focus on **minimizing false negatives** (missing a risky customer is the costly mistake).

## Dataset

- File: `Risk_Alert_Classifier_Dataset_4600 - Risk_Alert_Classifier_Dataset_4600.csv.csv`
- Records: 4600 customers
- Target: `risk_status` (about 88% safe, 12% risky, so the data is imbalanced)
- Dropped columns: `customer_id`, `last_transaction_date`

## Project Structure

| Part | Content |
|---|---|
| A | Conceptual understanding (theory) |
| B | Data preparation: train/test split, mode imputation, scaling, one-hot encoding, KNN imputation |
| C | Baseline model: Logistic Regression, confusion matrix, Type-I and Type-II errors |
| D | Imbalanced data: Under-Sampling, Over-Sampling, SMOTE, ADASYN and comparison |
| E | Tree-based models: Decision Tree and Random Forest |
| F | Hyperparameter tuning: Randomized Search CV, Grid Search CV, tuned vs untuned |
| G | ROC curves, AUC-ROC comparison, final model selection and final report |

## Models Used

- Logistic Regression (baseline)
- Decision Tree (untuned and tuned)
- Random Forest (untuned and tuned)

## Key Results (test set, 920 records)

| Model | False Negatives | False Positives | Recall | AUC-ROC |
|---|---|---|---|---|
| **Logistic Regression** | **0** | 1 | 1.0000 | 1.0000 |
| Decision Tree (untuned) | 22 | 16 | 0.8226 | 0.9055 |
| Decision Tree (tuned) | 15 | 9 | 0.8790 | 0.9861 |
| Random Forest (untuned) | 12 | 0 | 0.9032 | 0.9992 |
| Random Forest (tuned) | 7 | 0 | 0.9435 | 0.9998 |

**Final model: Logistic Regression.** It missed no risky customer and made only one false alarm.

## Tech Stack

Python, Streamlit, pandas, NumPy, scikit-learn, imbalanced-learn, matplotlib

## How to Run

1. Install the requirements:

   ```
   pip install streamlit pandas numpy scikit-learn imbalanced-learn matplotlib
   ```

2. Keep `project_3.py` and the CSV file in the same folder.

3. Start the app:

   ```
   streamlit run project_3.py
   ```
