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
# #NB Model (Here for posterity but not used in final report)
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
import shap
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import seaborn as sns

#========================================
#====Variable dictionary=================
#========================================

# title: title name given to the earthquake
# magnitude: The magnitude of the earthquake
# date_time: date and time
# cdi: The maximum reported intensity for the event range
# mmi: The maximum estimated instrumental intensity for the event
# alert: The alert level - “green”, “yellow”, “orange”, and “red”
# tsunami: "1" for events in oceanic regions and "0" otherwise
# sig: A number describing how significant the event is. Larger numbers indicate a more significant event. This value is determined on a number of factors, including: magnitude, maximum MMI, felt reports, and estimated impact
# net: The ID of a data contributor. Identifies the network considered to be the preferred source of information for this event.
# nst: The total number of seismic stations used to determine earthquake location.
# dmin: Horizontal distance from the epicenter to the nearest station
# gap: The largest azimuthal gap between azimuthally adjacent stations (in degrees). In general, the smaller this number, the more reliable is the calculated horizontal position of the earthquake. Earthquake locations in which the azimuthal gap exceeds 180 degrees typically have large location and depth uncertainties
# magType: The method or algorithm used to calculate the preferred magnitude for the event
# depth: The depth where the earthquake begins to rupture
# latitude / longitude: coordinate system by means of which the position or location of any place on Earth's surface can be determined and described
# location: location within the country
# continent: continent of the earthquake hit country
# country: affected country


#=============================================
#==============Data Cleaning==================
#=============================================

url=  ('earthquake_data.csv')
df_raw=pd.read_csv(url)
print(df_raw.isna().sum())
print(len(df_raw))
#The Continent, Alert, Country Columns are mostly null so we are going to drop them
df_raw= df_raw.drop(['alert','country', 'continent'], axis=1)

#With those Columns out, there are 5 null values with location, We can afford to drop those rows

df=df_raw.dropna()
#
print(df.isna().sum())
print(len(df))
#777 observations is still good enough to use ML techniques

#Let us remove outliers from out numerical columns

num=(['magnitude','cdi','mmi','sig','nst','dmin','gap', 'depth'])

def remove_outliers(df, columns):
    for col in columns:
        print('Working on column: {}'.format(col))

        mean = df[col].mean()
        sd = df[col].std()
        print(mean + 3*sd)
        df = df[(df[col] <= mean + (3 * sd))]
    return df
#
df=remove_outliers(df,num)

print(len(df))
#No Outliers were removed per the standard mean + 3*sd criteria

#Now we will save the cleaned dataset as a separate csv
df.to_csv('EQ_Clean.csv')


##EDA
# # ##Uploading clean database
# url=  ('https://raw.githubusercontent.com/apmejiac/DATS6202_FinalProject-Group1/main/earthquake_data.csv')
df=pd.read_csv('EQ_Clean.csv')
df_eda= df.copy()
df_eda = df_eda.drop(['Unnamed: 0'], axis=1)

##Splitting date time column to get further information on year, month and time
df_eda['date_time'] = pd.to_datetime(df_eda['date_time'])
df_eda['year'] = df_eda['date_time'].dt.year
df_eda['month'] = df_eda['date_time'].dt.month

#
#
##The EDA proccess was performed in a clean dataset
# Histogram distribution of the magnitude of earthquakes in the sample clean data
plt.hist(df['magnitude'], color = "teal", ec="powderblue")
plt.xlabel('Magnitude')
plt.title('Distribution of earthquake magnitude in the sample')
plt.tight_layout()
plt.show()

print(f'The major incidence of earthquake have an incidence of 6.5')
#
#Number of earthquakes per year
df_yr= df_eda.groupby(['year'])['year'].count()  ##df to count yearly incidences
df_yr=df_yr.to_frame()
df_yr.rename(columns = {'year':'count'}, inplace = True)
df_yr=df_yr.reset_index()
df_mnt= df_eda.groupby(['month'])['month'].count() ##df to count monthly incidences
df_mnt=df_mnt.to_frame()
df_mnt.rename(columns = {'month':'count'}, inplace = True)
df_mnt=df_mnt.reset_index()
#
#
fig, axs = plt.subplots(2, 1)
axs[0].plot(df_yr['year'], df_yr['count'], color= 'teal')
axs[0].set_title('Earthquake occurences per year')
axs[0].set_xlabel('Year')
axs[0].set_ylabel('Count')
axs[0].grid(True)
#
axs[1].plot(df_mnt['month'], df_mnt['count'],color= 'powderblue')
axs[1].set_title('Earthquake occurences per month')
axs[1].set_xlabel('Month')
axs[1].set_ylabel('Count')
axs[1].grid(True)
fig.tight_layout()
plt.show()

print(f'The years 2013 and 2015 and the month of November are the ones with more earthquake occurences')

# ## Countries with more count of earthquakes
#
##Filling country and city based on latitude and longitude--- requieres  pip install geopy----- left commented because takes alot of time and has alot of nan
# from tkinter import *
# from geopy.geocoders import Nominatim
#
# # initialize Nominatim API
# df_eda['latitude2'] = df_eda['latitude'].astype(str)
# df_eda['longitude2'] = df_eda['longitude'].astype(str)
# df_eda['Geolocation'] = df_eda[['latitude2', 'longitude2']].apply(lambda x: " ".join(x), axis=1)
# # df_eda.update(df_eda[['Geolocation']].applymap('"{}"'.format))
#
# geolocator= Nominatim(user_agent="MLG1")
# for i in df_eda.index:
#     try:
#         location = geolocator.reverse("{},{}".format(df_eda['latitude'][i], df_eda['longitude'][i]),timeout=None)
#         df_eda.loc[i,'location_address']= location.address
#     except:
#         df_eda.loc[i, 'location_address']= ""
#
# print(df_eda)
#
print(f'Many of the lat/long data points seems to be originating in the ocean for this reason location will be used and where missing values are present we are removing those values')

# # ### Splitting location column to be able to obtain country - city
# #
## ### Continent graph
# df_eda[['location']] = df_eda[['location']].fillna('nan,nan')
df_eda.loc[10,'location'] = " nan," + df_eda['location'][10]
df_eda.loc[28, 'location'] = " nan," + df_eda['location'][28]
df_eda.loc[35:36,'location'] = " nan," + df_eda['location']
df_eda.loc[53,'location'] = " nan," + df_eda['location'][53]
df_eda.loc[67,'location'] = " nan," + df_eda['location'][67]
df_eda.loc[72,'location'] = " nan," + df_eda['location'][72]
df_eda.loc[101,'location'] = " nan," + df_eda['location'][101]
df_eda.loc[104,'location'] = " nan," + df_eda['location'][104]
df_eda.loc[107,'location'] = " nan," + df_eda['location'][107]
df_eda.loc[120,'location'] = " nan," + df_eda['location'][120]
df_eda.loc[155,'location'] = " nan," + df_eda['location'][155]
df_eda.loc[161,'location'] = " nan," + df_eda['location'][161]
df_eda.loc[184,'location'] = " nan," + df_eda['location'][184]
df_eda.loc[235,'location'] = " nan," + df_eda['location'][235]
df_eda.loc[256,'location'] = " nan," + df_eda['location'][256]
df_eda.loc[260,'location'] = " nan," + df_eda['location'][260]
df_eda.loc[293,'location'] = " nan," + df_eda['location'][293]
df_eda.loc[310,'location'] = " nan," + df_eda['location'][310]
df_eda.loc[339,'location'] = " nan," + df_eda['location'][339]
df_eda.loc[345,'location'] = " nan," + df_eda['location'][345]
df_eda.loc[486,'location'] = " nan," + df_eda['location'][486]
df_eda.loc[493,'location'] = " nan," + df_eda['location'][493]
df_eda.loc[518,'location'] = " nan," + df_eda['location'][518]
df_eda.loc[565,'location'] = " nan," + df_eda['location'][565]
df_eda.loc[619,'location'] = " nan," + df_eda['location'][619]

# # Creating a new column with country after the comma in location column
df_eda['Country_2'] = df_eda['location'].str.rsplit(',').str[-1]
# #
# # ##Concatenating country names
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Alaska','United States of America')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('California','United States of America')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('CA','United States of America')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('India region','India')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Japan region','Japan')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('NV Earthquake','United States of America')
df_eda['Country_2'] = df_eda['Country_2'].str.replace(' New Zealand region',' New Zealand')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Russia region','Russia')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Washington','United States of America')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Ant','Chile')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Fiji region','Fiji')

df_eda['Country_2'] = df_eda['Country_2'].str.replace(' Costa Rica','Costa Rica')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Kermadec Islands region','Kermadec Islands')

df_eda['Country_2'] = df_eda['Country_2'].str.replace('Micronesia region','Micronesia')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Philippine Islands ','Philippines')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Philippine Islands region','Philippines')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('South Sandwich Islands region','South Sandwich Islands')

df_eda['Country_2'] = df_eda['Country_2'].str.replace('the Loyalty Islands ','Loyalty Islands')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('the Fiji Islands','Fiji')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('the Kermadec Islands','Kermadec Islands')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Idaho','United States of America')
df_eda.loc[518,'Country_2'] = "Indonesia"

# #
#
# ##Removing white space before string
df_eda['Country_2'] = df_eda['Country_2'].str.lstrip()

# # ##Plot of countries with most earthquakes
df_countries= df_eda.groupby(['Country_2']).aggregate({'Country_2':'count', 'magnitude':'mean'})  ##df to count yearly incidences
# df_countries= df_countries.to_frame()
df_countries.rename(columns = {'Country_2':'count'}, inplace = True)
df_countries= df_countries.reset_index()
#
#
##Top 20 countries
df_countries= df_countries.nlargest(20, "count")

##Plot
# Figure Size
fig, ax = plt.subplots(1,2,figsize=(16, 9), sharex= True, tight_layout=True)

# Horizontal Bar Plot
ax[0].barh(df_countries['Country_2'],df_countries['count'],color = "teal")

# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax[0].spines[s].set_visible(False)

# Remove x, y Ticks
ax[0].xaxis.set_ticks_position('none')
ax[0].yaxis.set_ticks_position('none')
#
# Add padding between axes and labels
ax[0].xaxis.set_tick_params(pad=5)
ax[0].yaxis.set_tick_params(pad=10)

# Add x, y gridlines
ax[0].grid(color='grey',
        linestyle='-.', linewidth=0.5,
        alpha=0.2)

# Show top values
ax[0].invert_yaxis()

# Add annotation to bars
# for i in ax[0].patches:
#     plt.text(i.get_width() + 0.2, i.get_y() + 0.5,
#              str(round((i.get_width()), 2)),
#              fontsize=10, fontweight='bold',
#              color='grey')

# Add Plot Title
ax[0].set_title('Top 20 countries with more earthquake occurrences',
             loc='left')

# Magnitude Plot
ax[1].barh(df_countries['Country_2'],df_countries['magnitude'],color = "powderblue")

# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax[1].spines[s].set_visible(False)

# Remove x, y Ticks
ax[1].xaxis.set_ticks_position('none')
ax[1].yaxis.set_ticks_position('none')

# Add padding between axes and labels
ax[1].xaxis.set_tick_params(pad=5)
ax[1].yaxis.set_tick_params(pad=10)

# Add x, y gridlines
ax[1].grid(color='grey',
        linestyle='-.', linewidth=0.5,
        alpha=0.2)

# Show top values
ax[1].invert_yaxis()

# Add annotation to bars
# for i in ax[1].patches:
#     plt.text(i.get_width() + 0.2, i.get_y() + 0.5,
#              str(round((i.get_width()), 2)),
#              fontsize=10, fontweight='bold',
#              color='grey')

# Add Plot Title
ax[1].set_title('Top 20 countries with greater Earthquake magnitude')
plt.show()



## Map of earthquakes #####colorbar rescalepending
import geopandas as gpd

# initialize an axis
fig, ax = plt.subplots(figsize=(15,10))

# plot map on axis
world = gpd.read_file(
               gpd.datasets.get_path("naturalearth_lowres"))
world.plot(color= "lightgrey", edgecolor='black', ax=ax)
# parse dates for plot's title
first_year = df_eda["date_time"].min().strftime("%b %Y")
last_year = df_eda["date_time"].max().strftime("%b %Y")

# plot points
df_eda.plot(x="longitude", y="latitude", kind="scatter", c='magnitude', s=10,
            colormap="YlOrRd",
            title=f"Earthquakes in the World {first_year} to {last_year}",
            ax=ax)
plt.show()


##Plotly interactive map
# import country_converter as coco
# cc = coco.CountryConverter()
# import pycountry_convert as pc
# df_eda.insert(0,"iso_alpha", " ")
# df_eda['iso_alpha'] = coco.convert(names=df_eda.Country_2.tolist(), to='ISO3', not_found=None)
#

df_plotly= df_eda.copy()
df_plotly.sort_values('year', inplace=True)
import plotly.express as px
lat= df.latitude
lon= df.longitude


fig = px.scatter_geo(df_plotly, lat= lat,lon= lon, color="magnitude", color_continuous_scale=["yellow",
   "orange", "red"], hover_name="Country_2", size= 100*(df_plotly["magnitude"]-df_plotly["magnitude"].min())/(df_plotly["magnitude"].max()-df_plotly["magnitude"].min()),
               animation_frame= 'year', projection="equirectangular", title= 'Earthquakes between 2001 to 2022 ')
fig.show(renderer= 'browser')


##Correlation matrix

import seaborn as sns

sns.set(font_scale=1.5)
plt.figure(figsize=(13,8))
sns.heatmap(df_eda.corr(),annot= True, annot_kws={"size": 7}, fmt=".2")#, cmap= 'YlGnBu')
plt.title('Correlation Coefficient between features- Original space')
plt.tight_layout()
plt.show()

#Scatter matrix
plt.figure(figsize=(13,8))
sns.pairplot(df_eda, hue= 'tsunami')
plt.show()

# # NUmber of earthquakes that led to tsunami
# #import plotly.express as px
# df = px.data.gapminder()
# fig = px.scatter(df, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
#            size="pop", color="continent", hover_name="country", facet_col="continent",
#            log_x=True, size_max=45, range_x=[100,100000], range_y=[25,90])
# fig.show()
#
# ----add continents----


# #
#
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


ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap=plt.cm.Blues)
plt.tight_layout()
plt.title('Confusion Matrix most accurate model Random Forest')
plt.tight_layout()
plt.show()

##Checking for accuracy, precision and recall

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
F1= f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)



print("Accuracy RF:", round(accuracy2,4))
print("Precision RF:", round(precision,4))
print("Recall RF:", round(recall,4))
print("F1 RF:", round(F1,4))
print("ROC_AUC RF:", round(roc_auc,4))

# Create a series containing feature importances from the model and feature names from the training data
feature_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# Plot a simple bar chart
feature_importances.plot.bar()
plt.tight_layout()
plt.show()

##Shap values visualization RF
import shap
explainer= shap.TreeExplainer(rf)
shap_v= explainer.shap_values(X_test)
shap.summary_plot(shap_v[0],X_test)
shap.summary_plot(shap_v,X_test)


# ## Trying out Ada Boost

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

dt= DecisionTreeClassifier(max_depth=1)

#Instantiate Ada Boost

ab = AdaBoostClassifier(base_estimator=dt, n_estimators=100)

#Fit training set
ab.fit(X_train,y_train)

#Hyper parameters
grid = {'n_estimators': [10, 50, 100, 500],
        'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0]}
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

#param grid
param_grid = GridSearchCV(estimator= ab,
                          param_grid = grid,
                          n_jobs=-1,
                          cv=cv,
                          scoring= 'accuracy')

# Fit the grid search object to the data
g_result= param_grid.fit(X_train, y_train)

# Create a variable for the best model
best_ada = g_result.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:', g_result.best_params_)

y_pred_pr = g_result.predict(X_test)
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_pr)


# Model Accuracy: how often is the classifier correct?
print("Accuracy AdaBoost:",round((metrics.accuracy_score(y_test, y_pred_pr)),4))
print("Precision AdaBoost:",round((metrics.precision_score(y_test, y_pred_pr)),4))
print("Recall AdaBoost:",round((metrics.recall_score(y_test, y_pred_pr)),4))
print("F1 AdaBoost:",round((f1_score(y_test, y_pred_pr)),4))
print("ROC_AUC AdaBoost:",round((roc_auc_score(y_test, y_pred_pr)),4))


cm = confusion_matrix(y_test, y_pred_pr)


ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap=plt.cm.Blues)
plt.tight_layout()
plt.title('Confusion Matrix most accurate model AdaBoost')
plt.tight_layout()
plt.show()

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





