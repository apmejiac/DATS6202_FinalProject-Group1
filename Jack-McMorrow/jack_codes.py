# Jack Codes
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
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix, accuracy_score, recall_score, roc_auc_score
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
        ("numeric", StandardScaler(), ["magnitude", "cdi", "mmi", "sig", "nst", "dmin", "gap", "depth", "latitude", "longitude"])
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
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))
print(confusion_matrix(y_test, knn.predict(X_test)))
print(classification_report(y_test, knn.predict(X_test)))
print(roc_auc_score(y_test, knn.predict(X_test)))
plot_confusion_matrix(knn, X_test, y_test, cmap=plt.cm.Blues)
plt.show()
# %%
ks = [k for k in range(3, 25, 2)]
scores = []
for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test)*100)
plt.plot(ks, scores)
plt.title("Accuracy of KNN at Different k's")
plt.xlabel("k")
plt.ylabel("Accuracy")
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
    #outcome = outcome.to_array()
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
    #outcome = outcome.to_array()
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
data = pd.read_csv("EQ_Clean.csv")
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
