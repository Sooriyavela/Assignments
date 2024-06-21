# %%
# 1 -  Clean and prepare your data:
# The data in this exercise have been simulated to mimic real, dirty data.
# Please clean the data with whatever method(s) you believe to be best/most suitable.
# You may create new features. However, you may not add or supplement with external data.

# %%
import shap
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')
# %%
df = pd.read_csv('/Users/sooriya/Documents/MLAlgorithms/data_project3.csv')

# %%
df.shape

# %%
df.info()

# %%
df.isnull().sum().sum()
# %%
# Removing duplicate rows if exists


def duplicate_rows_cleaning(dataframe):
    if dataframe.duplicated().sum() > 0:
        dataframe = dataframe.drop_duplicates()
    return dataframe


dataframe = df
print("The number of instances after removing duplicate rows includes:{0}".format(
    duplicate_rows_cleaning(dataframe).shape))
# %%
# drop the columns where
# 1.if the one of the value in a column counts for more than or equal to 95%.


def remove_columns(df):
    col_to_drop = []
    for col in df.columns:
        most_common_pct = df[col].value_counts(normalize=True).iloc[0]
        if most_common_pct >= 0.95:
            col_to_drop.append(col)
    print(col_to_drop)
    df = df.drop(columns=col_to_drop)
    return df


df = remove_columns(df)
df.shape
# %%
# cleaning the categorical columns.
contingency_table = pd.crosstab(df['x31'], df['x93'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
if p < 0.05:
    df = df.drop(columns=['x93'])

# %%
df.shape
# %%
# useof count encoding to replace categorical values to numerical ones.


def clean_categorical_col(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            col_dic = df[col].value_counts().to_dict()
            df[col] = df[col].map(col_dic)
    return df


df = clean_categorical_col(df)
y_data = df['y']
X_data = df.drop(columns=['y'])
# %%
# use of simple imputer mean to fill the null values.
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_data)
X_data = pd.DataFrame(X_imputed, columns=X_data.columns)

# %%
X_data.isnull().sum().sum()
# %%
# identifying the correlated predictors and removing one of them to avoid redundant data.
corr_matrix = X_data.corr(method='pearson', min_periods=1)
col_to_drop = set()
threshold_value = 0.7
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) >= threshold_value:
            # Get the name of one of the highly correlated columns
            colname = corr_matrix.columns[i]
            col_to_drop.add(colname)  # %%
X_data = X_data.drop(columns=col_to_drop)
# %%
X_data.info()

# %%
y_data.value_counts()

# %%
# using undersampling to nulify class imbalance
under_sampler = RandomUnderSampler(random_state=50)
X_data, y_data = under_sampler.fit_resample(X_data, y_data)

# %%
print(X_data.shape)
print(y_data.value_counts())
# %%
# standardizing the values
scaler = StandardScaler()
X_data = scaler.fit_transform(X_data)
# %%
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=50)

# %%
# 2-  Build your models:
# For this exercise, you are required to build five models. The first model must be a logistic regression.
# Then other 4 is the combination of bagging and boosting algorithms. Mimic the similar techniques we did during the class March 20th.
# For each model, find the top 3 most important features. Shap value has to be calculated for one of your model and explain it in briefly.

# 3- Evaluate your model:
# For each model calculate accuracy, precision, recall and AUC score. Graph AUC score as well.
# Create a table to make comparison for all models and decide which model is the best. State your reasoning.
# %%
# Linear regression Model
Model1 = LogisticRegression()
Model1.fit(X_train, y_train)
scores = cross_val_score(Model1, X_train, y_train, cv=5, scoring='accuracy')
# %%
np.mean(scores)
# %%
Pred_val1 = Model1.predict(X_test)
# %%
accuracy1 = accuracy_score(y_test, Pred_val1)
print(
    "The accuracy score of Logistic regression model is:{0}".format(accuracy1))
class_report = classification_report(y_test, Pred_val1)
print("The classification report of Logistic regression model is:")
print(class_report)
precision = precision_score(y_test, Pred_val1)
print(
    "The precision score of Logistic regression model is:{0}".format(precision))
recall = recall_score(y_test, Pred_val1)
print("The recall score of Logistic regression model is:{0}".format(recall))
y_pred_probs = Model1.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_probs)
print("the AUC Score is : {0}".format(auc_score))

# %%
feature_importance = abs(Model1.coef_[0])
feature_series = pd.Series(feature_importance)
sorted_features = feature_series.sort_values(ascending=False)
top_three_features = sorted_features[:3]
print("The top 3 features of Logistic regression model are : ")
print(top_three_features.index)
# %%
# creating a function to graph the roc curve for different models using roc score.


def roccurve(roc_auc):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.show()


# %%
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
roc_auc1 = auc(fpr, tpr)
print("The ROC Curve for Linear regression is as below:")
print(roccurve(roc_auc1))
# %%
# shap values graph for the linear regression model
explainer = shap.LinearExplainer(Model1, X_train)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
# The top features contribute more to the output variation than the bottom ones.
# Here, x6 appears to be the most impactful feature, followed by x37, x30, etc.
# This means that high values of x6 lead to higher predictions by the model,
# while lower values of x6 tend to lower the model's predictions.
# x6 has a wide spread, meaning its impact on the model's output varies a lot across different observations.
# %%
# Random forest Model
forest = RandomForestClassifier()
forest.fit(X_train, y_train)
scores = cross_val_score(forest, X_train, y_train, cv=5, scoring='accuracy')
print("the Accuracy score for the Trained data is  :{0}".format(
    np.mean(scores)))
Pred_val2 = forest.predict(X_test)
accuracy2 = accuracy_score(y_test, Pred_val2)
print(
    "The accuracy score of Random Forest model is:{0}".format(accuracy2))
class_report = classification_report(y_test, Pred_val2)
print("The classification report of Random Forest model is:")
print(class_report)
precision = precision_score(y_test, Pred_val2)
print(
    "The precision score of Random Forest model is:{0}".format(precision))
recall = recall_score(y_test, Pred_val2)
print("The recall score of Random Forest model is:{0}".format(recall))
y_pred_probs = forest.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_probs)
print("the AUC Score is : {0}".format(auc_score))
# %%
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
roc_auc2 = auc(fpr, tpr)
print("The ROC Curve for Random Forest model is as below:")
print(roccurve(roc_auc2))
# %%
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
print("Top three features in Random Forest model is:")
for i in indices[:3]:
    print(f"x{i}")
# %%
# adaboost classifier
ada_clf = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_clf.fit(X_train, y_train)
scores = cross_val_score(ada_clf, X_train, y_train, cv=5, scoring='accuracy')
print("the Accuracy score for the Trained data is  :{0}".format(
    np.mean(scores)))
Pred_val3 = ada_clf.predict(X_test)
accuracy3 = accuracy_score(y_test, Pred_val3)
print(
    "The accuracy score of Adaboost model is:{0}".format(accuracy3))
class_report = classification_report(y_test, Pred_val3)
print("The classification report of Adaboost model is:")
print(class_report)
precision = precision_score(y_test, Pred_val3)
print(
    "The precision score of Adaboost model is:{0}".format(precision))
recall = recall_score(y_test, Pred_val3)
print("The recall score of Adaboost model is:{0}".format(recall))
y_pred_probs = ada_clf.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_probs)
print("the AUC Score is : {0}".format(auc_score))

# %%
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
roc_auc3 = auc(fpr, tpr)
print("The ROC Curve for Adaboost model is as below:")
print(roccurve(roc_auc3))

# %%
importances = ada_clf.feature_importances_
indices = np.argsort(importances)[::-1]
print("Top three features in Adaboost model is:")
for i in indices[:3]:
    print(f"x{i}")
# %%
# catboost
catboost_model = CatBoostClassifier(
    iterations=100, learning_rate=0.1, depth=3, verbose=0)
catboost_model.fit(X_train, y_train)
scores = cross_val_score(catboost_model, X_train,
                         y_train, cv=5, scoring='accuracy')
print("the Accuracy score for the Trained data is  :{0}".format(
    np.mean(scores)))
Pred_val4 = catboost_model.predict(X_test)
accuracy4 = accuracy_score(y_test, Pred_val4)
print(
    "The accuracy score of catboost model is:{0}".format(accuracy4))
class_report = classification_report(y_test, Pred_val4)
print("The classification report of catboost model is:")
print(class_report)
precision = precision_score(y_test, Pred_val4)
print(
    "The precision score of catboost model is:{0}".format(precision))
recall = recall_score(y_test, Pred_val4)
print("The recall score of catboost model is:{0}".format(recall))
y_pred_probs = catboost_model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_probs)
print("the AUC Score is : {0}".format(auc_score))
# %%
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
roc_auc4 = auc(fpr, tpr)
print("The ROC Curve for catboost is as below:")
print(roccurve(roc_auc4))
# %%
importances = catboost_model.feature_importances_
indices = np.argsort(importances)[::-1]
print("Top three features in catboost model is:")
for i in indices[:3]:
    print(f"x{i}")
# %%
# XGBoost
xgboost_model = xgb.XGBClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=3, use_label_encoder=False, eval_metric='logloss')
xgboost_model.fit(X_train, y_train)
scores = cross_val_score(xgboost_model, X_train,
                         y_train, cv=5, scoring='accuracy')
print("the Accuracy score for the Trained data is  :{0}".format(
    np.mean(scores)))
Pred_val5 = xgboost_model.predict(X_test)
accuracy5 = accuracy_score(y_test, Pred_val4)
print(
    "The accuracy score of XGBoost model is:{0}".format(accuracy5))
class_report = classification_report(y_test, Pred_val5)
print("The classification report of XGBoost model is:")
print(class_report)
precision = precision_score(y_test, Pred_val5)
print(
    "The precision score of XGBoost model is:{0}".format(precision))
recall = recall_score(y_test, Pred_val5)
print("The recall score of XGBoost model is:{0}".format(recall))
y_pred_probs = xgboost_model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_probs)
print("the AUC Score is : {0}".format(auc_score))
# %%
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
roc_auc5 = auc(fpr, tpr)
print("The ROC Curve for XGBoost model is as below:")
print(roccurve(roc_auc5))

# %%
importances = xgboost_model.feature_importances_
indices = np.argsort(importances)[::-1]
print("Top three features in XGBoost model is:")
for i in indices[:3]:
    print(f"x{i}")
# %%
# Table involving roc scores
data = {"roc_score": [roc_auc1, roc_auc2, roc_auc3, roc_auc4, roc_auc5],
        "Accuracy": [accuracy1, accuracy2, accuracy3, accuracy4, accuracy5]
        }
index = ["Logistic Regression", "RandomForest",
         "AdaBoost", "CatBoost", "XGBoost"]
Table = pd.DataFrame(data, index=index)
# %%
Table
# From the table, XGBoost and CatBoost have the highest ROC scores (0.766633 and 0.765782, respectively),
# indicating they are better at distinguishing between the positive and negative classes in your dataset compared to the other models.
# AdaBoost and RandomForest show moderately high ROC scores, with Logistic Regression trailing slightly behind.
# This suggests that ensemble methods (which RandomForest, AdaBoost, CatBoost, and XGBoost are) tend to perform better for
# this particular task in terms of distinguishing between classes.

# RandomForest shows the highest accuracy (0.689492), closely followed by XGBoost and CatBoost (both at 0.686477).
# This suggests that, for correctly predicting the labels in your dataset,
# RandomForest performs slightly better, with XGBoost and CatBoost being very close contenders.
# %%
# 4- Summary of your findings / Conclusion


# From the performance of all the models, we can see that the features like x6, x3 and x37
# are the most important features for our classification.
# with respect to the model, both XGBoost and Catboost have been best preforming model in terms of better classification than others.

# %%
