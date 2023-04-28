#THIS FILE IS A COMPILATION OF MODELING DONE BY ALEX AND JACK
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
#=================================================================
#JACKS MODELING CODE (Which consists of 3 files)
#File 1: sklearnmodeling.py
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix, accuracy_score, recall_score
from sklearn.compose import ColumnTransformer

# %%
data = pd.read_csv("EQ_Clean.csv")

data.head()
# %%
# Feature Engineering: Extract country
for i, row in data.iterrows():
    data_country = row["location"].split(',')
    if len(data_country) == 2:
        data.loc[i, "country"] = data_country[1].strip()
    else:
        data.loc[i, "country"] = data_country[0]

print(data.country.value_counts())

# Preparing Data
data_features = data.drop(["title", "date_time", "tsunami", "location"], axis=1)
data_target = data[["tsunami"]]
data_features.head()
# %%
# IMPORTANT: The magnitude rictor scale is an exponential scale, so it might not make sense
# to scale this variable as it might mess it up. Thoughts?
dataPreprocessor = ColumnTransformer(transformers=
[
    ("categorical", OneHotEncoder(), ["magType", "net", "country"]),
    ("numeric", StandardScaler(),
     ["magnitude", "cdi", "mmi", "sig", "nst", "dmin", "gap", "depth", "latitude", "longitude"])
], verbose_feature_names_out=False, remainder="passthrough")

data_features_newmatrix = dataPreprocessor.fit_transform(data_features)
new_col_names = dataPreprocessor.get_feature_names_out()
data_features_new = pd.DataFrame(data_features_newmatrix.toarray(), columns=new_col_names)
print(data_features_new.shape)
data_features_new.head()
# %%
# Train test split
X_train, X_test, y_train, y_test = train_test_split(data_features_new, data_target, test_size=0.25, random_state=342)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# %%
# knn
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))
print(confusion_matrix(y_test, knn.predict(X_test)))
print(classification_report(y_test, knn.predict(X_test)))
# %%
ks = [k for k in range(3, 25, 2)]
scores = []
for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
plt.plot(ks, scores)
plt.show()
# %%
mlp = MLPClassifier()
mlp.fit(X_train, y_train)
print(mlp.score(X_test, y_test))
print(confusion_matrix(y_test, mlp.predict(X_test)))
print(classification_report(y_test, mlp.predict(X_test)))
# %% Trying out different activation functions
funcs = ['relu', 'identity', 'logistic', 'tanh']
for func in funcs:
    mlp = MLPClassifier(activation=func)
    mlp.fit(X_train, y_train)
    print(f"Model score for {func} function: {mlp.score(X_test, y_test)}")
# %% Trying different solvers
solvers = ['lbfgs', 'sgd', 'adam']
for solver in solvers:
    mlp = MLPClassifier(solver=solver, activation='relu')
    mlp.fit(X_train, y_train)
    print(f"Model score for {solver} solver: {mlp.score(X_test, y_test)}")
# %% relu and adam are the best
mlp_optum = MLPClassifier(activation='relu', solver="adam")
mlp_optum.fit(X_train, y_train)
print(mlp_optum.score(X_test, y_test))
print(classification_report(y_test, mlp_optum.predict(X_test)))
matrix = plot_confusion_matrix(mlp_optum, X_test, y_test, cmap=plt.cm.Blues)
plt.show()

# %%
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
print(rfc.score(X_test, y_test))
print(confusion_matrix(y_test, rfc.predict(X_test)))
print(classification_report(y_test, rfc.predict(X_test)))
# %% finding good cutoff values
predicted_probs = rfc.predict_proba(X_test)
cutoffs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
scores = []
for cutoff in cutoffs:
    outcome = []
    for prediction in predicted_probs[:, 1]:
        if prediction >= cutoff:
            outcome.append(1)
        else:
            outcome.append(0)
    # outcome = outcome.to_array()
    score = accuracy_score(y_test, outcome)
    scores.append(score)
    print(f"Accuracy score for {cutoff} cutoff: {score}")

plt.plot(cutoffs, scores)
plt.title("Accuracy of Random Forest Classifier with Different Cutoff Values")
plt.xlabel("Cutoff Value")
plt.ylabel("Accuracy Score")
plt.show()
# %% DOing the same for recall, which we might want to priortize
scores = []
for cutoff in cutoffs:
    outcome = []
    for prediction in predicted_probs[:, 1]:
        if prediction >= cutoff:
            outcome.append(1)
        else:
            outcome.append(0)
    # outcome = outcome.to_array()
    score = recall_score(y_test, outcome)
    scores.append(score)
    print(f"Recall score for {cutoff} cutoff: {score.__round__(2)}")

plt.plot(cutoffs, scores)
plt.title("Recall of Random Forest Classifier with Different Cutoff Values")
plt.xlabel("Cutoff Value")
plt.ylabel("Recall Score")
plt.show()
# %%
from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(X_train, y_train)
print(classification_report(y_test, xgb.predict(X_test)))
matrix = plot_confusion_matrix(xgb, X_test, y_test, cmap=plt.cm.Blues)
plt.show()
# %%
from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, y_train)
print(svc.score(X_test, y_test))
print(confusion_matrix(y_test, svc.predict(X_test)))
print(classification_report(y_test, svc.predict(X_test)))
# %%
#==============================================================
#JACK'S other File
#kears_modeling.py
# Keras Networks
# -------------------------------------------------
# Importing Libraries
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import make_scorer, accuracy_score
# --------------------------------------------------
# Importing Data
data = pd.read_csv("/content/EQ_Clean.csv")
print(data.shape)
print(data.head())
data.drop(['title', 'date_time', 'location'], axis=1, inplace=True)
print
(data.head())
# --------------------------------------------------
# Transforming and Splitting Data
preprocessor = ColumnTransformer(transformers=
    [
        ("categorical", OneHotEncoder(), ["net", "magType"]),
        ("numeric", StandardScaler(), ["magnitude", "cdi", "mmi", "sig", "dmin", "gap", "depth"])
    ], verbose_feature_names_out=False, remainder="passthrough"
)

data_transform = preprocessor.fit_transform(data)
new_col_names = preprocessor.get_feature_names_out()
data_new = pd.DataFrame(data_transform, columns=new_col_names)
data_new.head()

target = data["tsunami"]
features = data_new.drop(["tsunami"], axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=22)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# --------------------------------------------------
# Building first MLP
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(26,)))
# model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=513, epochs=100, verbose=0, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
# --------------------------------------------------
# First Grid Search
# Defining function to make model
def create_nn(optimizer="adam", activation="relu", neurons=100, batch_size=513, epochs=100):
  model = Sequential()
  model.add(Dense(neurons, activation=activation, input_shape=(26,)))
  model.add(Dense(neurons, activation=activation))
  model.add(Dense(1, activation="sigmoid"))

  model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

  history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(X_test, y_test))
  score = model.evaluate(X_test, y_test, verbose=0)

  return model

# Wrapping model to scikit-learn
model = KerasClassifier(build_fn=create_nn, verbose=0)

# Making parameter grid
param_grid = {
    "optimizer": ["adam", "sgd"],
    "activation": ["relu", "tanh"],
    "neurons": [25, 50, 100, 150, 200],
    "batch_size": [32, 64, 128],
    "epochs": [10, 20, 100]
}

# Define the scoring function
scoring = {'accuracy': make_scorer(accuracy_score)}

# Perform the grid search
grid_search = GridSearchCV(model, param_grid=param_grid, cv=2, scoring=scoring, refit='accuracy', verbose=2)
grid_result = grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters: %s" % grid_result.best_params_) # Output: Best parameters: {'activation': 'tanh', 'batch_size': 32, 'epochs': 100, 'neurons': 150, 'optimizer': 'adam'}
print("Best score: %f" % grid_result.best_score_) # Output: Best score: 0.908363
# --------------------------------------------------
# Second Grid Search
param_grid = {
    "optimizer": ["adam"],
    "activation": ["tanh"],
    "neurons": [125, 150, 175],
    "batch_size": [8, 16, 32, 48],
    "epochs": [100, 150, 200]
}

# Define the scoring function
scoring = {'accuracy': make_scorer(accuracy_score)}

# Perform the grid search
grid_search = GridSearchCV(model, param_grid=param_grid, cv=2, scoring=scoring, refit='accuracy', verbose=2)
grid_result = grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters: %s" % grid_result.best_params_) # Best parameters: {'activation': 'tanh', 'batch_size': 48, 'epochs': 200, 'neurons': 150, 'optimizer': 'adam'}
print("Best score: %f" % grid_result.best_score_) # Best score: 0.947349
# --------------------------------------------------
# Final Model
model = Sequential()
model.add(Dense(150, activation='tanh', input_shape=(26,)))
model.add(Dense(150, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=48, epochs=200, verbose=0, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

#ALEJANDRAS MODELING
import time

import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import seaborn as sns

# # Modelling
# ##Random forest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# # Tree Visualisation
from sklearn.tree import export_graphviz
# from IPython.display import Image
# # import graphviz
#
# Splitting the data into features (X) and target (y)
print(df_eda.isna().sum())
#
###Changing variables from categorical to dummies
df_mod= df_eda.copy()
df_mod['net'].replace(['us','at','ak', 'pt','nn','ci','hv','nc','official','duputel','uw'], [1,2,3,4,5,6,7,8,9,10,11], inplace=True)
df_mod['magType'].replace(['mww','mwb','Mi','ml','mw','mwc','ms','mb','md'],[1,2,3,4,5,6,7,8,9], inplace=True)
X = df_mod.drop(['tsunami','title','location','date_time','year', 'month', 'Country_2'], axis=1)
y = df_mod['tsunami']

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

##Fitting and evaluating the Model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
## predict
y_pred = rf.predict(X_test)

rf.estimators_
#
# fn=X.columns
# # cn=Y.columns
# fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (15,10), dpi=800)
# tree.plot_tree(rf.estimators_[0],
#                feature_names = fn,
#                class_names='tsunami',
#                filled = True);
# # fig.savefig('rf_individualtree.png')
# plt.show()

##Model evaluation

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy*100,2))


#Hyper parameters
param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

# Create a random forest classifier
# rf = RandomForestClassifier()

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf,
                                 param_distributions = param_dist,
                                 n_iter=5,
                                 cv=5)

# Fit the random search object to the data
rand_search.fit(X_train, y_train)

# Create a variable for the best model
best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)

# Generate predictions with the best model
y_pred = best_rf.predict(X_test)

##Testing acaccutacy with best model
accuracy2 = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy2*100,2))

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.tight_layout()
plt.title('Confusion Matrix most accurate model')
plt.show()

##Checking for accuracy, precision and recall


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", round(accuracy2,2))
print("Precision:", round(precision,2))
print("Recall:", round(recall,2))


# Create a series containing feature importances from the model and feature names from the training data
feature_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# Plot a simple bar chart
feature_importances.plot.bar()
plt.tight_layout()
plt.show()

# ## Trying out Ada Boost

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

dt= DecisionTreeClassifier(max_depth=1)

#Instantiate Ada Boost

ab = AdaBoostClassifier(base_estimator=dt, n_estimators=100)

#Fit training set
ab.fit(X_train,y_train)

y_pred_pr = ab.predict_proba(X_test)[:,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_pr)

##Evaluate test- set
ab_roc=roc_auc_score(y_test, y_pred_pr)
print('ROC AUC score: {:.2f}'.format(ab_roc))

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.tight_layout()
plt.show()

#### SVM Modelling

from sklearn.model_selection import train_test_split
from sklearn import svm
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  ##random state?

##Generating the mdoel
svm_clf= svm.SVC( kernel='linear')
svm_clf.fit(X_train, y_train)
y_pred = svm_clf.predict(X_test)


# Model Accuracy: how often is the classifier correct?
print("Accuracy SVC:",metrics.accuracy_score(y_test, y_pred))

print("Precision SVC:",metrics.precision_score(y_test, y_pred))
print("Recall SVC:",metrics.recall_score(y_test, y_pred))

##Tuning Hyperparameters
svm_clf_r= svm.SVC(kernel='poly')
svm_clf_r.fit(X_train, y_train)
y_pred = svm_clf_r.predict(X_test)


# Model Accuracy: how often is the classifier correct?
print("Accuracy SVC poly:",metrics.accuracy_score(y_test, y_pred))

print("Precision SVC poly:",metrics.precision_score(y_test, y_pred))
print("Recall SVC poly:",metrics.recall_score(y_test, y_pred))

svm_clf= svm.SVC(C= 0.95, kernel='linear', gamma= 'auto')
svm_clf.fit(X_train, y_train)
y_pred = svm_clf.predict(X_test)


# Model Accuracy: how often is the classifier correct?
print("Accuracy SVC:",metrics.accuracy_score(y_test, y_pred))

print("Precision SVC:",metrics.precision_score(y_test, y_pred))
print("Recall SVC:",metrics.recall_score(y_test, y_pred))




