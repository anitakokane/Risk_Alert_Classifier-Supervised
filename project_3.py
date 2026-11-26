import pandas as pd 
import numpy as np
import streamlit as st
from sklearn.impute import KNNImputer


st.set_page_config(page_title="Risk Alert classifier")

st.title("Risk Alert Classifier",text_alignment="center")
st.text("The Objective of this project is to design,evaluate,and optimize a classification " \
"system that predicts high-risk customer behaviour.")

st.markdown("### Part B:Dataset Understanding & Preparation")

from pathlib import Path
df = pd.read_csv("risk_data.csv")
st.markdown("#### Dataset Preview:")
st.dataframe(df)


## X y Split 
x=df.drop(["customer_id","last_transaction_date","risk_status"],axis=1)
st.write("Input Features:")
st.dataframe(x)

y=df["risk_status"]
st.write("Target Variable :")
st.dataframe(y)


## Train Test split 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
st.success("Data is splitted into train and test dataset.")
st.write("Shape of x_train:",x_train.shape)
st.write("Shape of x_test:",x_test.shape)
st.write("Shape of y_train:",y_train.shape)
st.write("Shape of y_test:",y_test.shape)

## Identifying missing values 
st.write("Missing Values in train data:",x_train.isnull().sum())
st.write("Missing Values in test data:",x_test.isnull().sum())
categorical=x_train.select_dtypes(include="object").columns
for col in categorical:
    x_train[col] = x_train[col].fillna(x_train[col].mode()[0])
    x_test[col]=x_test[col].fillna(x_train[col].mode()[0])
st.success("Categorical Values Imputed by MODE")


## Scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
num=x_train.select_dtypes(include="number").columns
x_train[num]=scaler.fit_transform(x_train[num])
x_test[num]=scaler.transform(x_test[num])

## encoding
from sklearn.preprocessing import OneHotEncoder

cat = x_train.select_dtypes(include="object").columns
encoder = OneHotEncoder(
     handle_unknown="ignore", sparse_output=False
)
encoder.set_output(transform="pandas")  

x_train_enc = encoder.fit_transform(x_train[cat])
x_test_enc = encoder.transform(x_test[cat])

x_train = pd.concat([x_train.drop(columns=cat), x_train_enc], axis=1)
x_test = pd.concat([x_test.drop(columns=cat), x_test_enc], axis=1)

st.write("Train data after Scaling and encoding:",x_train)
st.write("Test data after Scaling and encoding:",x_test)

## KNN Imputer
 
numerical=x_train.select_dtypes(include="number").columns
imputer=KNNImputer(n_neighbors=5)
x_train[numerical] =imputer.fit_transform(x_train[numerical])
x_test[numerical]=imputer.transform(x_test[numerical])
st.success("KNN succesfully applied")

st.write("After Imputation (Train):",x_train.isnull().sum())
st.write("After Imputation (Test):", x_test.isnull().sum())

st.markdown("### Part C:Baseline Classification Model")
## Logistic Regression 
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
st.write("Confusion Matrix:",confusion_matrix(y_test,y_pred))
st.write("Accuracy Score:",accuracy_score(y_test,y_pred))
st.write("Precision Score:",precision_score(y_test,y_pred))
st.write("Recall Score:",recall_score(y_test,y_pred))
st.write("F1 Score:",f1_score(y_test,y_pred))
st.write("AUC ROC score:",roc_auc_score(y_test,lr.predict_proba(x_test)[:,1]))

st.markdown("#### Type-I errors")
st.text("A Type-I error happens when the model wrongly predics the positive " \
" Class even though the actual class is negative.It is like 'false alarm'." \
" For example,a healthy person being told that he has disease." \
"In our matrix, the number of Type-I errors is 1.")

st.markdown("#### Type-II errors")
st.text("A Type-II error happens when the model wrongly predicts the" \
"negative class even though the actual class is positive. It is like a" \
"'missed detection'. For example, a sick person being told that he is healthy." \
"In our matrix, the number of Type-II errors is 0.")

st.markdown("### Part-D Handling Imbalanced Data")

counts = y_train.value_counts()
st.write("Class distribution:", counts)
st.write("Percentage:", y_train.value_counts(normalize=True) * 100)

st.markdown("#### Impact of class imbalanced on model performance:")
st.text("-The model sees class 0 very often and class 1 very rarely, so it " \
"becomes biased towards class 0.")
st.text("-Accuracy looks high but is misleading. Even a model that predicts “0”"
" for everyone would get about 88% accuracy without finding a single class 1 (risky)" \
" sample.")
st.text("-Recall and F1-Score of the minority class become low, because many risky " \
"cases get predicted as safe (False Negatives).")
st.text("-In risk prediction, missing a risky case is costly, so this bias is a " \
"serious problem.")

from imblearn.under_sampling import RandomUnderSampler
## Under Sampling 

st.markdown("#### Under-Sampling")
rus=RandomUnderSampler(random_state=42)
x_train_us,y_train_us=rus.fit_resample(x_train,y_train)

st.write("Class distribution after Under-Sampling:",y_train_us.value_counts())

model_us = LogisticRegression(max_iter=1000)
model_us.fit(x_train_us, y_train_us)

pred_us = model_us.predict(x_test)
prob_us = model_us.predict_proba(x_test)[:, 1]

st.write("Confusion Matrix:",confusion_matrix(y_test,pred_us))
st.write("Accuracy Score:",accuracy_score(y_test,pred_us))
st.write("Precision Score:",precision_score(y_test,pred_us))
st.write("Recall (minority):", recall_score(y_test, pred_us))
st.write("F1-Score:", f1_score(y_test, pred_us))
st.write("AUC-ROC:", roc_auc_score(y_test, prob_us))

## Over Sampling 

st.markdown("#### Over Sampling")
from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=42)
x_train_os,y_train_os=ros.fit_resample(x_train,y_train)

st.write("Class distribution after Over-Sampling:",y_train_os.value_counts())

model_os = LogisticRegression(max_iter=1000)
model_os.fit(x_train_os, y_train_os)

pred_os = model_os.predict(x_test)
prob_os = model_os.predict_proba(x_test)[:, 1]

st.write("Confusion Matrix:",confusion_matrix(y_test,pred_os))
st.write("Accuracy Score:",accuracy_score(y_test,pred_os))
st.write("Precision Score:",precision_score(y_test,pred_os))
st.write("Recall (minority):", recall_score(y_test, pred_os))
st.write("F1-Score:", f1_score(y_test, pred_os))
st.write("AUC-ROC:", roc_auc_score(y_test, prob_os))

## SMOTE 
st.markdown("#### Synthetic Minority Over-Sampling Technique(SMOTE)")

from imblearn.over_sampling import SMOTE
smote=SMOTE(random_state=42)

x_train_sm, y_train_sm = smote.fit_resample(x_train, y_train)

st.write("Class distribution after SMOTE:")
st.write(y_train_sm.value_counts())

model_sm = LogisticRegression(max_iter=1000)
model_sm.fit(x_train_sm, y_train_sm)

pred_sm = model_sm.predict(x_test)
prob_sm = model_sm.predict_proba(x_test)[:, 1]

st.write("Confusion Matrix:",confusion_matrix(y_test,pred_sm))
st.write("Accuracy Score:",accuracy_score(y_test,pred_sm))
st.write("Precision Score:",precision_score(y_test,pred_sm))
st.write("Recall (minority):", recall_score(y_test, pred_sm))
st.write("F1-Score:", f1_score(y_test, pred_sm))
st.write("AUC-ROC:", roc_auc_score(y_test, prob_sm))

## ADASYN

st.markdown("#### Adaptive Synthetic Sampling(ADASYN)")
from imblearn.over_sampling import ADASYN
adasyn = ADASYN(random_state=42)
x_train_ad, y_train_ad = adasyn.fit_resample(x_train, y_train)

st.write("Class distribution after ADASYN:")
st.write(y_train_ad.value_counts())

model_ad = LogisticRegression(max_iter=1000)
model_ad.fit(x_train_ad, y_train_ad)

pred_ad = model_ad.predict(x_test)
prob_ad = model_ad.predict_proba(x_test)[:, 1]

st.write("Confusion Matrix:",confusion_matrix(y_test,pred_ad))
st.write("Accuracy Score:",accuracy_score(y_test,pred_ad))
st.write("Precision Score:",precision_score(y_test,pred_ad))
st.write("Recall (minority):", recall_score(y_test, pred_ad))
st.write("F1-Score:", f1_score(y_test, pred_ad))
st.write("AUC-ROC:", roc_auc_score(y_test, prob_ad))

##15 comparison 

st.markdown("#### Comparison")

table = pd.DataFrame({
    "Technique": ["Before Balancing", "Under-Sampling", "Over-Sampling", "SMOTE", "ADASYN"],
    "Recall": [1.0000, 1.0000, 1.0000, 0.9919, 1.0000],
    "F1-Score": [0.9960, 0.9650, 0.9688, 0.9919, 0.9764],
    "AUC-ROC": [0.9994, 1.0000, 1.0000, 1.0000, 0.9999],
})

st.dataframe(table.style.format({
    "Recall": "{:.4f}",
    "F1-Score": "{:.4f}",
    "AUC-ROC": "{:.4f}",
}))

st.markdown("#### Conclusion:")
st.text("The baseline model already performed very well (Recall = 1.0," \
" F1 = 0.9960, AUC-ROC = 0.9994),so balancing did not improve the results. " \
"Under-Sampling, Over-Sampling and ADASYN kept recall at 1.0 but increased " \
"false positives,which lowered the F1-Score. SMOTE had a slightly lower recall "
"(0.9919) because it missed one risky case.AUC-ROC improved only slightly after " \
"balancing (0.9994 to about 1.0), so the model separates the classes very well in " \
"every case. Among the balancing techniques, ADASYN gave the best F1-Score(0.9764).")

st.markdown("### Tree- Based Classification Models")
from sklearn.tree import DecisionTreeClassifier

st.markdown("#### Decision Tree Classifier")
tree=DecisionTreeClassifier(max_depth=4,random_state=42)
tree.fit(x_train,y_train)

pred_t=tree.predict(x_test)
prob_t = tree.predict_proba(x_test)[:, 1]

st.write("Confusion Matrix:")
st.dataframe(confusion_matrix(y_test, pred_t))

st.write("Accuracy Score:", accuracy_score(y_test, pred_t))
st.write("Precision Score:", precision_score(y_test, pred_t))
st.write("Recall Score:", recall_score(y_test, pred_t))
st.write("F1 Score:", f1_score(y_test, pred_t))
st.write("AUC ROC score:", roc_auc_score(y_test, prob_t))

## Checking overfitting 
st.markdown("#### Train and Test Accuracy")
st.write("Train Accuracy:",tree.score(x_train,y_train))
st.write("Test Accuracy:",tree.score(x_test,y_test))

##Random Forest Classifier 
from sklearn.ensemble import RandomForestClassifier

st.markdown("#### Random Forest Classifier")
forest=RandomForestClassifier(n_estimators=100,max_depth=5,random_state=42)
forest.fit(x_train,y_train)

pred_f=forest.predict(x_test)
prob_f = forest.predict_proba(x_test)[:, 1]

st.write("Confusion Matrix:")
st.dataframe(confusion_matrix(y_test, pred_f))

st.write("Accuracy Score:", accuracy_score(y_test, pred_f))
st.write("Precision Score:", precision_score(y_test, pred_f))
st.write("Recall Score:", recall_score(y_test, pred_f))
st.write("F1 Score:", f1_score(y_test, pred_f))
st.write("AUC ROC score:", roc_auc_score(y_test, prob_f))

## Checking overfitting 
st.markdown("#### Train and Test Accuracy")
st.write("Train Accuracy:",forest.score(x_train,y_train))
st.write("Test Accuracy:",forest.score(x_test,y_test))

## Comparison 
st.markdown("#### Comaprison between Decision Tree and Random Forest")

st.text("Accuracy: Random Forest has higher test accuracy (98.70% vs 95.87%) "
"and better F1-Score (0.9492 vs 0.8430) and AUC-ROC (0.9992 vs 0.9055).")
st.text("Generalization: Decision Tree has a train-test gap of about 2.5% "
"(98.40% train vs 95.87% test), so it overfits more. Random Forest has a gap of "
"only about 0.5% (99.21% train vs 98.70% test), so it generalizes better on unseen data.")
st.text("Random Forest combines many trees trained on random samples of data and " \
"features, and takes a majority vote. This reduces variance and overfitting. " \
"A single tree is sensitive to small changes in the data.")
st.text("Conclusion: Random Forest is better than Decision Tree in both accuracy"
" and generalization. Decision Tree is preferred only when easy interpretation"
" is needed.")

st.markdown("#### Part F : Hyperparameter Tuning ")
from sklearn.model_selection import RandomizedSearchCV

st.markdown("#### Decision Tree")
dt_params = {
    "max_depth": [3, 5, 7, 10, 15, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "criterion": ["gini", "entropy"],
}

dt_search = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_params,
    n_iter=20,
    cv=5,
    scoring="f1",
    random_state=42,
    n_jobs=1,
)
dt_search.fit(x_train, y_train)

st.write("Best Parameters:", dt_search.best_params_)
st.write("Best CV F1-Score:", dt_search.best_score_)

tuned_dt = DecisionTreeClassifier(
        max_depth=7,
        min_samples_split=20,
        min_samples_leaf=8,
        criterion="entropy",
        random_state=42)
tuned_dt.fit(x_train,y_train)
st.markdown("#### Tune Decision Tree Performance ")

pred_td=tuned_dt.predict(x_test)
prob_td = tuned_dt.predict_proba(x_test)[:, 1]

st.write("Confusion Matrix:")
st.dataframe(confusion_matrix(y_test, pred_td))

st.write("Accuracy Score:", accuracy_score(y_test, pred_td))
st.write("Precision Score:", precision_score(y_test, pred_td))
st.write("Recall Score:", recall_score(y_test, pred_td))
st.write("F1 Score:", f1_score(y_test, pred_td))
st.write("AUC ROC score:", roc_auc_score(y_test, prob_td))
    
st.markdown("#### Random Forest")
rf_params = {
    "n_estimators": [50, 100, 150, 200, 300],
    "max_depth": [5, 8, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}

rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    n_iter=20,
    cv=5,
    scoring="f1",
    random_state=42,
    n_jobs=1,
)
rf_search.fit(x_train, y_train)

st.write("Best Parameters:", rf_search.best_params_)
st.write("Best CV F1-Score:", rf_search.best_score_)

tuned_rf =RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="log2",
        random_state=42)
tuned_rf.fit(x_train,y_train)
st.markdown("#### Tune Random Forest Performance")

pred_rf=tuned_rf.predict(x_test)
prob_rf = tuned_rf.predict_proba(x_test)[:, 1]

st.write("Confusion Matrix:")
st.dataframe(confusion_matrix(y_test, pred_rf))

st.write("Accuracy Score:", accuracy_score(y_test, pred_rf))
st.write("Precision Score:", precision_score(y_test, pred_rf))
st.write("Recall Score:", recall_score(y_test, pred_rf))
st.write("F1 Score:", f1_score(y_test, pred_rf))
st.write("AUC ROC score:", roc_auc_score(y_test, prob_rf))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (confusion_matrix, precision_score,
                             recall_score, f1_score, roc_auc_score)

st.markdown("### Q21: Grid Search CV (Random Forest)")

param_grid = {
    "n_estimators": [150, 200, 250],
    "max_depth": [10, 15, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=1,
)
grid.fit(x_train, y_train)

st.write("Best Parameters:", grid.best_params_)
st.write("Best CV F1-Score:", round(grid.best_score_, 4))

# Evaluate the fine-tuned model on test data
best_rf = grid.best_estimator_
pred = best_rf.predict(x_test)
prob = best_rf.predict_proba(x_test)[:, 1]

st.write("Confusion Matrix:")
st.dataframe(pd.DataFrame(confusion_matrix(y_test, pred)))

st.write("Train Accuracy:", best_rf.score(x_train, y_train))
st.write("Test Accuracy:", best_rf.score(x_test, y_test))
st.write("Precision Score:", precision_score(y_test, pred))
st.write("Recall Score:", recall_score(y_test, pred))
st.write("F1 Score:", f1_score(y_test, pred))
st.write("AUC ROC score:", roc_auc_score(y_test, prob))

# Comparison with untuned Random Forest
pf = forest.predict(x_test)
table = pd.DataFrame({
    "Model": ["Untuned Random Forest", "Grid Search Random Forest"],
    "Test Acc": [accuracy_score(y_test, pf), best_rf.score(x_test, y_test)],
    "Precision": [precision_score(y_test, pf), precision_score(y_test, pred)],
    "Recall": [recall_score(y_test, pf), recall_score(y_test, pred)],
    "F1-Score": [f1_score(y_test, pf), f1_score(y_test, pred)],
    "AUC-ROC": [roc_auc_score(y_test, forest.predict_proba(x_test)[:, 1]), roc_auc_score(y_test, prob)],
})
st.dataframe(table.round(4))

models = {
    "Decision Tree (Untuned)": tree,
    "Decision Tree (Tuned)": tuned_dt,
    "Random Forest (Untuned)": forest,
    "Random Forest (Tuned)": tuned_rf,
}

rows = []
for name, model in models.items():
    pred = model.predict(x_test)                
    prob = model.predict_proba(x_test)[:, 1]
    rows.append([name,
                 accuracy_score(y_test, pred),
                 precision_score(y_test, pred),
                 recall_score(y_test, pred),
                 f1_score(y_test, pred),
                 roc_auc_score(y_test, prob)])

table = pd.DataFrame(rows, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"])
st.dataframe(table.round(4))

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

st.markdown("### Part G: Model Evaluation and ROC Analysis")
st.markdown("#### Q23: ROC Curve for All Models")

models = {
    "Logistic Regression": lr,
    "Decision Tree (Untuned)": tree,
    "Decision Tree (Tuned)": tuned_dt,
    "Random Forest (Untuned)": forest,
    "Random Forest (Tuned)": tuned_rf,
}

fig, ax = plt.subplots(figsize=(7, 5))

for name, model in models.items():
    prob = model.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")

ax.plot([0, 1], [0, 1], "k--", label="Random Guess")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate (Recall)")
ax.set_title("ROC Curve - All Models")
ax.legend(loc="lower right")
st.pyplot(fig)

# Compare AUC-ROC scores
rows = []
for name, model in models.items():
    prob = model.predict_proba(x_test)[:, 1]
    rows.append([name, roc_auc_score(y_test, prob)])

table = pd.DataFrame(rows, columns=["Model", "AUC-ROC"])
table = table.sort_values("AUC-ROC", ascending=False)   # best model on top
st.dataframe(table.round(4))


## 25 
st.text("The business requirement is to minimize false negatives, because a risky "
        "customer predicted as safe is the costly mistake. Logistic Regression missed "
        "no risky customers (FN = 0, recall = 1.0) with only 1 false alarm, while the "
        "tree-based models missed several. Therefore Logistic Regression is the best "
        "final model.")

st.markdown("### Part H: Final Report")

# 1. Best model
st.markdown("#### 1. Best Classification Model and Justification")
st.markdown("""
**Final model: Logistic Regression.**
- It missed **no risky customer** (False Negatives = 0, Recall = 1.0) out of 124 in the test data.
- It made only **1 false alarm** (False Positive = 1), so Precision = 0.992 and F1-Score = 0.996.
- AUC-ROC is about 1.0, so it separates risky and safe customers almost perfectly.
- Accuracy is 99.89%, and the model is simple and easy to explain to the business.
- The best tree model (tuned Random Forest) still missed 7 risky customers, so it was not chosen,
  because the business requirement is to minimize false negatives.
""")

# 2. Impact of imbalance handling
st.markdown("#### 2. Impact of Imbalance Handling Techniques")
st.markdown("""
The data is imbalanced: **88.23% safe (class 0)** and **11.77% risky (class 1)**.
Under-Sampling, Over-Sampling, SMOTE and ADASYN were applied on the training data only.
- The baseline model already had Recall = 1.0, so balancing gave **no improvement**.
- Under-Sampling, Over-Sampling and ADASYN kept Recall at 1.0 but increased false positives
  (6 to 9 compared with 1), which lowered Precision and F1-Score.
- SMOTE kept both errors low (1 FP and 1 FN) but lost a little recall (0.9919).
- The imbalance is mild and the classes are easy to separate, so balancing had very little effect.
""")

# 3. Comparison of metrics
st.markdown("#### 3. Comparison of Performance Metrics")
report = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree (Untuned)", "Decision Tree (Tuned)",
              "Random Forest (Untuned)", "Random Forest (Tuned)"],
    "Accuracy": [0.9989, 0.9587, 0.9739, 0.9870, 0.9924],
    "Recall": [1.0000, 0.8226, 0.8790, 0.9032, 0.9435],
    "F1-Score": [0.9960, 0.8430, 0.9008, 0.9492, 0.9709],
    "AUC-ROC": [1.0000, 0.9055, 0.9861, 0.9992, 0.9998],
    "False Negatives": [0, 22, 15, 12, 7],
    "False Positives": [1, 16, 9, 0, 0],
})
st.dataframe(report)
st.markdown("""
- Random Forest is better than a single Decision Tree, and tuning improved both tree models.
- Logistic Regression has the highest Recall and the lowest False Negatives.
""")

# 4. Business interpretation
st.markdown("#### 4. Business Interpretation of False Positives and False Negatives")
st.markdown("""
- **False Negative (Type-II error):** a risky customer is predicted as safe. The company may
  approve credit or ignore warning signs, and can face defaults and financial loss.
  **This is the costly mistake**, so we minimize it.
- **False Positive (Type-I error):** a safe customer is predicted as risky. The company does an
  extra check or may lose a good customer. This is a smaller, recoverable cost.
- Since one missed risky customer costs much more than one extra review, the model with
  zero false negatives is preferred, even if it makes a few false alarms.
""")

st.markdown("#### Conclusion")
st.markdown("""
Logistic Regression is recommended as the final model because it catches every risky customer
with only one false alarm. Accuracy alone is not enough on imbalanced data, so Recall, F1-Score
and AUC-ROC were used. Imbalance handling did not improve the strong baseline here.
""")

