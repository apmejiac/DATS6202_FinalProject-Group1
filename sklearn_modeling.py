# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
# %%
data = pd.read_csv("EQ_Clean.csv")

data.head()
# %%
# Preparing Data
data_features = data.drop(["title", "date_time", "location", "tsunami"], axis=1)
data_target = data[["tsunami"]]
data_features.head()
# %%
# IMPORTANT: The magnitude rictor scale is an exponential scale, so it might not make sense
# to scale this variable as it might mess it up. Thoughts?
dataPreprocessor = ColumnTransformer(transformers=
    [
        ("categorical", OneHotEncoder(), ["magType", "net"])
        # ("numeric", StandardScaler(), []) decide what to do with these
    ], verbose_feature_names_out=False, remainder="passthrough")

data_features_newmatrix = dataPreprocessor.fit_transform(data_features)
new_col_names = dataPreprocessor.get_feature_names_out()
data_features_new = pd.DataFrame(data_features_newmatrix, columns=new_col_names)
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
mlp = MLPClassifier(activation="logistic")
mlp.fit(X_train, y_train)
print(mlp.score(X_test, y_test))
print(confusion_matrix(y_test, mlp.predict(X_test)))
print(classification_report(y_test, mlp.predict(X_test)))
# %%
