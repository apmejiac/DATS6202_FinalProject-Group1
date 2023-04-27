import numpy as np
import pandas as pd
from scipy.stats import shapiro
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.stats
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


#Dropping Unneeded Features
df=pd.read_csv('EQ_Clean.csv')
df= df.drop(['title','date_time', 'location'], axis=1)
df=df.drop(df.columns[0], axis=1)

#Correlation
plt.matshow(df.corr())
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);
plt.show()

#We can see very low correlation, but lets also do some t-tests for correlation with
#Our target variable: tsunami
y=df['tsunami'].values
df=df.drop(['tsunami'], axis=1)
def corr_coeff( x,y):
    '''
    :param x: an array of numbers
    :param y: an array of numbers
    :return: the correlation between x and y
    num is the numerator
    denx is the x side of the denominator in the sqrt
    deny is the y side of the denominator in the sqrt
    den is the final denominator post sqrt
    '''
    num=0
    denx=0
    deny=0

    x_b= np.mean(x)
    y_b= np.mean(y)
    for i in range (0, len(x)):
        num+= (x[i]-x_b)*(y[i]-y_b)
        dx= (x[i]-x_b)
        dy= (y[i]-y_b)
        denx+=np.power(dx,2 )
        deny+=np.power(dy, 2 )
    total= round(num/(np.sqrt((denx)*(deny))),2)
    return(total)
def corr_t(D,a,b, alpha):
    '''
    :param D: a data frame
    :param a: an array of numbers or dataframe array
    :param b: an array of numbers or a dataframe column
    :return:
    '''
    n=len(D)
    r = corr_coeff(a,b)
    t_0= (r*((n-2)**0.5))/((1-(r**2))**0.5)
    p=scipy.stats.t.sf((t_0), n-3)
    if p < alpha:
        return("p=",p,"The correlation is significant, reject the null hypothesis")
    else:
        return("p=",p,"The correlation is not siginficant, fail to reject.")

def corr_calc (df):
    for i in (df.columns):
        print(f"{i}:",corr_t(df, y, df[i], 0.05))

#Dataframe of only numeric variables for calculations
num_df = df.select_dtypes(include=['int64','float64']).copy()
corr_calc(num_df)

#Norm Check
for i in num_df.columns:
    print(f"{i}:",shapiro(num_df[i]))

#We can see that the data is non-normal in all columns (due to low p-values) so let's
#We will use standard scaler to normalize the data

#Encoding Data
cat_df = df.select_dtypes(include=['object']).copy()
df[cat_df.columns] = df[cat_df.columns].astype(str)
df = pd.get_dummies(df)
OC=df.columns

#Scaling Data
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)

x= scaled_data
df=pd.DataFrame(x)

df.columns=OC
#Place the output values back in the prepared dataset
df['tsunami']=y
df.to_csv('DF_PREPPED.csv')


