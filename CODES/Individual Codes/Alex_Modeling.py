import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
df=pd.read_csv('DF_PREPPED.csv.')
x=df.drop(['tsunami'], axis=1)
x=x.drop(x.columns[0], axis=1)
y= df['tsunami'].values


#Train Test Split:
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

from sklearn.neural_network import MLPClassifier

#Initializing the multilayer perceptron
mlp = MLPClassifier(hidden_layer_sizes=10,solver='sgd',learning_rate_init= 0.01, max_iter=500)

mlp.fit(X_train, y_train)


from sklearn.metrics import confusion_matrix
y_pred = mlp.predict(X_test)
from sklearn.metrics import plot_confusion_matrix
#Pretty CM Graph
color = 'white'
matrix = plot_confusion_matrix(mlp, X_test, y_test, cmap=plt.cm.Blues)
plt.xlabel('Predicted Label', color=color)
plt.ylabel('True Label', color=color)
plt.title('MLP Confusion Matrix')
plt.gcf().axes[0].tick_params(colors=color)
plt.gcf().axes[1].tick_params(colors=color)
plt.show()
#
# #NB Model
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
#NB CM
color = 'white'
matrix = plot_confusion_matrix(gnb, X_test, y_test, cmap=plt.cm.Blues)
plt.title('GNB Confusion Matrix')
plt.xlabel('Predicted Label', color=color)
plt.ylabel('True Label', color=color)
plt.gcf().axes[0].tick_params(colors=color)
plt.gcf().axes[1].tick_params(colors=color)
plt.show()
#
print("MLP:",mlp.score(X_test, y_test))
print("GNB:",gnb.score(X_test, y_test))

#TSME
#Try RF
#PCA not necessary
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import  accuracy_score
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
print(model.score(X_test, y_test))
#First xgb cm
color = 'white'
matrix = plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues)
plt.title('Initial XGB Confusion Matrix')
plt.xlabel('Predicted Label', color=color)
plt.ylabel('True Label', color=color)
plt.gcf().axes[0].tick_params(colors=color)
plt.gcf().axes[1].tick_params(colors=color)
plt.show()


from sklearn.model_selection import GridSearchCV
# #Hyperparameter Tuning
param_grid = {
    "max_depth": [3, 4, 5, 7],
    "learning_rate": [0.1, 0.01, 0.05],
    "gamma": [0, 0.25, 1],
    "reg_lambda": [0, 1, 10],
    "scale_pos_weight": [1, 3, 5],
    "subsample": [0.8],
    "colsample_bytree": [0.5]}
#Init classifier
xgb_cl = xgb.XGBClassifier(objective="binary:logistic")

#Init Grid Search
grid_cv = GridSearchCV(xgb_cl, param_grid, n_jobs=-1, cv=3, scoring="roc_auc")

# Fit

_ = grid_cv.fit(X_train, y_train)
print("Grid CV1 best score:",grid_cv.best_score_)

print(grid_cv.best_params_)
#New Parameter ranges since some are at their end
param_grid2 = {
    "max_depth": [7,8,15, 20],
    "learning_rate": [0.005, 0.009, 0.01],
    "gamma": [0.25],
    "reg_lambda": [0.25, 0.5, 0.75, 1, 2],
    "scale_pos_weight": [3],
    "subsample": [0.8],
    "colsample_bytree": [0.5]
}

grid_cv_2 = GridSearchCV(xgb_cl, param_grid2,
                         cv=3, scoring="roc_auc", n_jobs=-1)

_ = grid_cv_2.fit(X_train, y_train)
print("gcv2 best score:",grid_cv_2.best_score_)
#Final XGB classifier
final_cl = xgb.XGBClassifier(
    **grid_cv_2.best_params_,
    objective="binary:logistic",
)

_ = final_cl.fit(X_train, y_train)
print("FINAL XGB Accuracy", final_cl.score(X_test, y_test))
from sklearn.metrics import roc_auc_score
A= roc_auc_score(y_test, final_cl.predict_proba(X_test)[:, 1])
B=roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
C= roc_auc_score(y_test, mlp.predict_proba(X_test)[:, 1])
D=roc_auc_score(y_test, gnb.predict_proba(X_test)[:, 1])
print("Tuned XGB ROC_AUC",A)
print("First XGB ROC_AUC", B)
print("MLP ROC_AUC",C)
print("GNB ROC_AUC", D)

color = 'white'
matrix = plot_confusion_matrix(final_cl, X_test, y_test, cmap=plt.cm.Blues)
plt.title('Final XGB Confusion Matrix')
plt.xlabel('Predicted Label', color=color)
plt.ylabel('True Label', color=color)
plt.gcf().axes[0].tick_params(colors=color)
plt.gcf().axes[1].tick_params(colors=color)
plt.show()

#SVM TIME
from sklearn import svm
SVM1 = svm.SVC(probability=True)
SVM1.fit(X_train, y_train)
E=roc_auc_score(y_test, SVM1.predict_proba(X_test)[:, 1])
print(E)

#Time for parameter Tuning
param_grid = {
    "C": [0.5, 1, 2, 3],
    "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
    "degree": [1, 3, 4,5],
    "gamma": [0.1, 0.5, 0.75],
    "probability": [True]}

grid_cv1 = GridSearchCV(SVM1, param_grid, n_jobs=-1, cv=3, scoring="roc_auc")


_ = grid_cv1.fit(X_train, y_train)
print("Grid CV1 best score:",grid_cv1.best_score_)

print(grid_cv1.best_params_)

#Time for a second Grid CV
param_grid = {
    "C": [2],
    "kernel": ['rbf'],
    "degree": [1],
    "gamma": [0.01, 0.025,0.075, 0.1],
    "probability": [True]}

grid_cv2 = GridSearchCV(SVM1, param_grid, n_jobs=-1, cv=3, scoring="roc_auc")


_ = grid_cv2.fit(X_train, y_train)
print("Grid CV2 best score:",grid_cv2.best_score_)

print(grid_cv2.best_params_)

SVMF = svm.SVC(probability=True, C=2, degree=1, gamma=0.025, kernel='rbf')
SVMF.fit(X_train, y_train)
F=roc_auc_score(y_test, SVMF.predict_proba(X_test)[:, 1])
print(F)
color = 'white'
matrix = plot_confusion_matrix(SVMF, X_test, y_test, cmap=plt.cm.Blues)
plt.title('Tuned SVM Confusion Matrix')
plt.xlabel('Predicted Label', color=color)
plt.ylabel('True Label', color=color)
plt.gcf().axes[0].tick_params(colors=color)
plt.gcf().axes[1].tick_params(colors=color)
plt.show()