import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px

df=pd.read_csv('EQ_Clean.csv')
#Original Dataset in case we want to save dat_time
DF=df.copy()
#Dropping non-useful variables
df= df.drop(['title','date_time', 'location'], axis=1)
# #Encoding Categorical Variables
cat_df = df.select_dtypes(include=['object']).copy()
df[cat_df.columns] = df[cat_df.columns].astype(str)
df = pd.get_dummies(df)
#Normalizing Data
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)
#PCA
pca = PCA(n_components=28)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
print(scaled_data.shape,x_pca.shape)
#Graph to see how many Components are needed
exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
fig=px.area(
    x=range(1, exp_var_cumul.shape[0] + 1),
    y=exp_var_cumul)
fig.update_layout(xaxis_title='# of Component', yaxis_title='Explained Variance')
fig.show()
# We need 20 Components to explain .97 of varaince, thus dropping 8 dimensions
pca = PCA(n_components=20)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
print(scaled_data.shape,x_pca.shape)
#Setting pca_df as to get it ready for modeling
df_pca= x_pca

