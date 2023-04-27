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