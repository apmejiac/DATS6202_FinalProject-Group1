import time

import numpy as np
import pandas as pd
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

url=  ('https://raw.githubusercontent.com/apmejiac/DATS6202_FinalProject-Group1/main/earthquake_data.csv')
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
# ##Uploading clean database
url=  ('https://raw.githubusercontent.com/apmejiac/DATS6202_FinalProject-Group1/main/earthquake_data.csv')
df=pd.read_csv(url)
df_eda= df.copy()

##Splitting date time column to get further information on year, month and time
df_eda['date_time'] = pd.to_datetime(df_eda['date_time'])
df_eda['year'] = df_eda['date_time'].dt.year
df_eda['month'] = df_eda['date_time'].dt.month



##The EDA proccess was performed in a clean dataset
# Histogram distribution of the magnitude of earthquakes in the sample clean data
plt.hist(df['magnitude'], color = "teal", ec="powderblue")
plt.xlabel('Magnitude')
plt.title('Distribution of earthquake magnitude in the sample')
plt.tight_layout()
plt.show()

print(f'The major incidence of earthquake have an incidence of 6.5')

#Number of earthquakes per year
df_yr= df_eda.groupby(['year'])['year'].count()  ##df to count yearly incidences
df_yr=df_yr.to_frame()
df_yr.rename(columns = {'year':'count'}, inplace = True)
df_yr=df_yr.reset_index()
df_mnt= df_eda.groupby(['month'])['month'].count() ##df to count monthly incidences
df_mnt=df_mnt.to_frame()
df_mnt.rename(columns = {'month':'count'}, inplace = True)
df_mnt=df_mnt.reset_index()


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

## Countries with more count of earthquakes

##Filling country and city based on latitude and longitude--- requieres  pip install geopy----- left commented because takes alot of time and has alot of nan
from tkinter import *
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

# ### Splitting location column to be able to obtain country - city



### Continent graph
df_eda[['location']] = df_eda[['location']].fillna('nan,nan')
df_eda.loc[5:6,'location'] = " nan," + df_eda['location']
df_eda.loc[15, 'location'] = " nan," + df_eda['location'][15]
df_eda.loc[22,'location'] = " nan," + df_eda['location'][22]
df_eda.loc[28,'location'] = " nan," + df_eda['location'][28]
df_eda.loc[30,'location'] = " nan," + df_eda['location'][30]
df_eda.loc[41,'location'] = " nan," + df_eda['location'][41]
df_eda.loc[45:46,'location'] = " nan," + df_eda['location']
df_eda.loc[49,'location'] = " nan," + df_eda['location'][49]
df_eda.loc[50,'location'] = " nan," + df_eda['location'][50]
df_eda.loc[52,'location'] = " nan," + df_eda['location'][52]
df_eda.loc[56:57,'location'] = " nan," + df_eda['location']
df_eda.loc[67,'location'] = " nan," + df_eda['location'][67]
df_eda.loc[73,'location'] = " nan," + df_eda['location'][73]
df_eda.loc[78:79,'location'] = " nan," + df_eda['location']
df_eda.loc[85,'location'] = " nan," + df_eda['location'][85]
df_eda.loc[88,'location'] = " nan," + df_eda['location'][88]
df_eda.loc[97,'location'] = " nan," + df_eda['location'][97]
df_eda.loc[102,'location'] = " nan," + df_eda['location'][102]
df_eda.loc[137,'location'] = " nan," + df_eda['location'][137]
df_eda.loc[140,'location'] = " nan," + df_eda['location'][140]
df_eda.loc[144,'location'] = " nan," + df_eda['location'][144]
df_eda.loc[160:161,'location'] = " nan," + df_eda['location']
df_eda.loc[186,'location'] = " nan," + df_eda['location'][186]
df_eda.loc[208,'location'] = " nan," + df_eda['location'][208]
df_eda.loc[214,'location'] = " nan," + df_eda['location'][214]
df_eda.loc[238,'location'] = " nan," + df_eda['location'][238]
df_eda.loc[242,'location'] = " nan," + df_eda['location'][242]
df_eda.loc[245,'location'] = " nan," + df_eda['location'][245]
df_eda.loc[301,'location'] = " nan," + df_eda['location'][301]
df_eda.loc[325,'location'] = " nan," + df_eda['location'][325]
df_eda.loc[329:330,'location'] = " nan," + df_eda['location']
df_eda.loc[335,'location'] = " nan," + df_eda['location'][335]
df_eda.loc[363,'location'] = " nan," + df_eda['location'][363]
df_eda.loc[365:367,'location'] = " nan," + df_eda['location']
df_eda.loc[374,'location'] = " nan," + df_eda['location'][374]
df_eda.loc[386,'location'] = " nan," + df_eda['location'][386]
df_eda.loc[389,'location'] = " nan," + df_eda['location'][389]
df_eda.loc[392:393,'location'] = " nan," + df_eda['location']
df_eda.loc[419,'location'] = " nan," + df_eda['location'][419]
df_eda.loc[425,'location'] = " nan," + df_eda['location'][425]
df_eda.loc[440:441,'location'] = " nan," + df_eda['location']
df_eda.loc[448,'location'] = " nan," + df_eda['location'][448]
df_eda.loc[464,'location'] = " nan," + df_eda['location'][464]
df_eda.loc[490,'location'] = " nan," + df_eda['location'][490]
df_eda.loc[566,'location'] = " nan," + df_eda['location'][566]
df_eda.loc[576,'location'] = " nan," + df_eda['location'][576]
df_eda.loc[583,'location'] = " nan," + df_eda['location'][583]
df_eda.loc[609,'location'] = " nan," + df_eda['location'][609]
df_eda.loc[611,'location'] = " nan," + df_eda['location'][611]
df_eda.loc[614,'location'] = " nan," + df_eda['location'][614]
df_eda.loc[659,'location'] = " nan," + df_eda['location'][659]
df_eda.loc[668:669,'location'] = " nan," + df_eda['location']
df_eda.loc[693,'location'] = " nan," + df_eda['location'][693]
df_eda.loc[715,'location'] = " nan," + df_eda['location'][715]
df_eda.loc[507,'location'] = " delta," + "Mexico"
df_eda.loc[668,'location'] = " nan," + "Indonesia"
df_eda.loc[693,'location'] = " nan," + "Afghanistan"
df_eda.loc[576,'location'] = " nan," + "Chile"
df_eda.loc[583,'location'] = " nan," + "Chile"
df_eda.loc[102,'location'] = " nan," + "Indonesia"
df_eda.loc[419,'location'] = " nan," + "Indonesia"
df_eda.loc[659,'location'] = " nan," + "Indonesia"

# # Creating a new column with country after the comma in location column
df_eda['Country_2'] = df_eda['location'].str.rsplit(',').str[-1]
#
# ##Concatenating country names
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Alaska','United States of America')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('California','United States of America')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('CA','United States of America')

df_eda['Country_2'] = df_eda['Country_2'].str.replace('India region','India')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Japan region','Japan')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('NV Earthquake','United States of America')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Russia region','Russia')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Washington','United States of America')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Bouvet Island region','Bouvet Island ')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Fiji region','Fiji')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Chile',' Chile')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Afghanistan',' Afghanistan')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Costa Rica',' Costa Rica')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Kermadec Islands region','Kermadec Islands')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Mauritius - Reunion region','Mauritius')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Mexico',' Mexico')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Micronesia region','Micronesia')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Philippine Islands region','Kermadec Islands')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('S','Indonesia')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('United Indonesiatates of America','United States of America')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Scotia Sea','United Kingdom')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Vanuatu region','Vanuatu')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('the Fiji Islands','Fiji')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('the Kermadec Islands','Kermadec Islands')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Indonesiaolomon Islands','Solomon Islands')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('El Indonesiaalvador','El Salvador')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Flores Indonesiaea','Indonesia')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Indonesiaamoa','Samoa')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Indonesiacotia Indonesiaea','United Kingdom')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Indonesiaouth Indonesiaandwich Islands','United Kingdom')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Indonesiaouth Indonesiahetland Islands','South Shetland Islands')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Indonesiavalbard and Jan Mayen','Svalbard and Jan Mayen')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Molucca Indonesiaea','Indonesia')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('United Kingdom region','United Kingdom')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('Idaho','United States of America')
df_eda['Country_2'] = df_eda['Country_2'].str.replace('off the west coast of northern Indonesiaumatra','Sumatra')


##Removing white space before string
df_eda['Country_2'] = df_eda['Country_2'].str.lstrip()

##Removing 5 NA
df_eda = df_eda[df_eda["Country_2"].str.contains("nan") == False]

##Plot of countries with most earthquakes
df_countries= df_eda.groupby(['Country_2']).aggregate({'Country_2':'count', 'magnitude':'mean'})  ##df to count yearly incidences
# df_countries= df_countries.to_frame()
df_countries.rename(columns = {'Country_2':'count'}, inplace = True)
df_countries= df_countries.reset_index()


##Top 20 countries
df_countries= df_countries.nlargest(20, "count")

##Plot
# Figure Size
fig, ax = plt.subplots(1,2,figsize=(16, 9))

# Horizontal Bar Plot
ax[0].barh(df_countries['Country_2'],df_countries['count'],color = "teal")

# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax[0].spines[s].set_visible(False)

# Remove x, y Ticks
ax[0].xaxis.set_ticks_position('none')
ax[0].yaxis.set_ticks_position('none')

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
for i in ax[0].patches:
    plt.text(i.get_width() + 0.2, i.get_y() + 0.5,
             str(round((i.get_width()), 2)),
             fontsize=10, fontweight='bold',
             color='grey')

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
for i in ax[1].patches:
    plt.text(i.get_width() + 0.2, i.get_y() + 0.5,
             str(round((i.get_width()), 2)),
             fontsize=10, fontweight='bold',
             color='grey')

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

# NUmber of earthquakes that led to tsunami



##Modeling

#Original Dataset in case we want to save dat_time
# DF=df.copy()
# #Dropping non-useful variables
# df= df.drop(['title','date_time', 'location'], axis=1)
# # #Encoding Categorical Variables
# cat_df = df.select_dtypes(include=['object']).copy()
# df[cat_df.columns] = df[cat_df.columns].astype(str)
# df = pd.get_dummies(df)
# #Normalizing Data
# scaler = StandardScaler()
# scaler.fit(df)
# scaled_data = scaler.transform(df)
# #PCA
# pca = PCA(n_components=28)
# pca.fit(scaled_data)
# x_pca = pca.transform(scaled_data)
# print(scaled_data.shape,x_pca.shape)
# #Graph to see how many Components are needed
# exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
# fig=px.area(
#     x=range(1, exp_var_cumul.shape[0] + 1),
#     y=exp_var_cumul)
# fig.update_layout(xaxis_title='# of Component', yaxis_title='Explained Variance')
# fig.show()
# # We need 20 Components to explain .97 of varaince, thus dropping 8 dimensions
# pca = PCA(n_components=20)
# pca.fit(scaled_data)
# x_pca = pca.transform(scaled_data)
# print(scaled_data.shape,x_pca.shape)
# #Setting pca_df as to get it ready for modeling
# df_pca= x_pca
