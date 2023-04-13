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
    print(f"Recall score for {cutoff} cutoff: {score}")
    
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
