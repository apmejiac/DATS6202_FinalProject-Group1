import numpy as np
import pandas as pd

df_raw=pd.read_csv('earthquake_data.csv')
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
#No Outliers were removed per the standard meand + 3*sd criteria

#Now we will save the cleaned dataset as a separate csv
df.to_csv('EQ_Clean.csv')