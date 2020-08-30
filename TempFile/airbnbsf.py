# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

abnb = pd.read_csv('AbnbSF_clean.csv')

abnb.shape

# abnb.info()

# abnb['Neighbourhood Cleansed'].value_counts().sort_index()

# combine.nunique().sort_values()
# abnb.select_dtypes(exclude='number').nunique().sort_values()

# abnb.select_dtypes(include='number').nunique().sort_values()

# abnb.info()

abnb.isnull().sum()

def wrangle(X):
    """Wrangle train, validate, and test sets in the same way"""
    
    # Prevent SettingWithCopyWarning
    X = X.copy()

    # Customise columns: Replace white space with underscore
    X.columns = X.columns.str.replace(' ', '_')  

    # Extract any 'good' Nulls value before they're removed.

    # 1. For string: Replace the nulls with 'not_listed', and impute missing values later. !!!
    # cols_NaNs1 = ['Amenities', 'Features', 'Host_Response_Time']
    # for col in cols_NaNs1:
    #     X[col] = X[col].fillna('not_listed')


    # 2. For numbers: Replace the nulls with 0 (fillna(0)), and impute missing values later. !!!



    # Fillna for other columns with most frquent values. Definitely with 1 (one)
    cols3 = ['Bedrooms', 'Beds', 'Bathrooms']
    for col in cols3:
        X[col] = X[col].fillna(X[col].mode()[0])


    # Drop duplicates



    # Edit some string on Zipcode


    # return the wrangled dataframe
    return X

abnb = wrangle(abnb)

# Rename dataframe from now on.

df = abnb.rename(columns={'Neighbourhood_Cleansed': 'Neighbourhood'})
df.shape

# df.info()

# Feature Engineering based on column 'Amenities'
# Does the unit provide TV, Internet, Breakfast, Kitchen, Air conditioning, Heating, Washer, Dryer


df.select_dtypes(include='number').nunique().sort_values()
df.select_dtypes(exclude='number').nunique().sort_values()
# df.dtypes

df['Price'].describe()

# Check Price percentile



# Make columns cleaner

# Make sure the target is clean


####################################################################

# Limit the price: remove 0.
df =   df[(df['Price'].between(20,800))]# 

# Limit with only the data with notNull >6500
df = df.dropna(thresh=6500, axis=1)
######################################################################


# Get a dataframe with all train columns except unneeded columns
# df_features = df.drop(columns=['ID', 'Scrape_ID', 'Host_ID', 'Latitude', 'Longitude'], axis=1) ## Without Lat/Long was done on 07/28
# df_features = df.drop(columns=['ID', 'Scrape_ID', 'Host_ID'], axis=1) ## Without Lat/Long was done on 07/28

# Get a list of the numeric features
# numeric_features = df_features.select_dtypes(include='number').columns.tolist()

# # Cardinality of the nonnumeric features
# cardinality = df_features.select_dtypes(exclude='number').nunique()

# # List of all categorical features with  (2< cardinality <70)
# cat_features = cardinality[(cardinality <= 70) & (cardinality > 2) ].index.tolist()

# # Combine the lists 
# features = numeric_features + cat_features
# print(features)

train = df[features]

print(train.shape)
train.head(2)


"""##Visualisation with Plotly. <br>What is the average Airbnb renting price per night in SF?"""

# Commented out IPython magic to ensure Python compatibility.
# Import packages for cgarting

import plotly.figure_factory as ff
import numpy as np
import chart_studio
import chart_studio.plotly as py
import plotly.express as px
# %matplotlib inline
import matplotlib.pyplot as plt

chart_studio.tools.set_credentials_file(username='henrygultom', api_key='TCczP1KyN9o3BdysTcBk')

# Show the Room_type distribution

fig = px.histogram(df, x='Price', title= 'Price and Room_type Distributions', nbins=40, color='Room_Type', width=1000, height=400)
fig.update_layout(legend=dict( yanchor="top",  y=.95,  xanchor="right", x=.95 ))
fig.update_layout(
    yaxis_title="Count",
    font=dict( family="Courier New, monospace", size=18))#, color="black")) # or color="RebeccaPurple"

fig.show()



# fig.write_html("histogram3.html")
# py.plot(fig, filename = 'price-roomtype', auto_open=True)

# df['Neighbourhood'].value_counts()

# Bedrooms in Neighborhood distribution

fig = px.histogram(df, x="Neighbourhood", title="Neighborhood and #Bedrooms Distributions" ,  nbins=40, color='Bedrooms', width=1000, height=500).update_xaxes(categoryorder="total descending")

fig.update_layout(legend=dict( yanchor="top",  y=.95,  xanchor="right", x=.95 ))
fig.update_layout(
    yaxis_title="Count",
    font=dict( family="Courier New, monospace", size=18))#, color="black")) # or color="RebeccaPurple"

# plt.xlabel("Neighborhood", size=20);
# plt.ylabel("Count", size=20);
fig.show();



# fig.write_html("histogram3.html")
# py.plot(fig, filename = 'nbrhood-bedroom', auto_open=True)

import seaborn as sns
import matplotlib.pyplot as plt



# plt.figure(figsize=(18, 6))
# chart2 = sns.boxplot(x = df["Neighbourhood"], y = df["Price"])#, palette="Blues");
# chart2.set_xticklabels(chart2.get_xticklabels(), rotation=90);
# plt.xlabel("Neighborhood", size=20)
# plt.ylabel("Price", size=20)
# plt.show();

# Pricec distribution across Neighborhood



# fig = px.box(df, x="Neighbourhood", y="Price", title = 'Price Distribution Across Neighborhood')
# fig.show()

# py.plot(fig, filename = 'nbrhood-price', auto_open=True)

# Number of people can be accommodated and price

# fig = px.box(df, x="Accommodates", y="Price", title = 'Price Distributions over #Accommodated Guests')

# fig.show()
# py.plot(fig, filename = 'accommodates-price', auto_open=True)

# Same graph with scatter plot

# fig = px.scatter(df, x="Accommodates", y="Price", trendline='ols', width=1000, height=400)
# fig.show()

# Accommodates distribution

# fig = px.histogram(df, x="Accommodates",  title="#Beds in 'Accommodates' Distribution" ,  nbins=40, color='Beds', width=1000, height=500) #.update_xaxes(categoryorder="total descending")

# fig.update_layout(legend=dict( yanchor="top",  y=.95,  xanchor="right", x=.95 ))


# plt.xlabel("Neighborhood", size=20);
# plt.ylabel("Count", size=20);
# fig.show();

# fig.write_html("histogram3.html")
# py.plot(fig, filename = 'accomodates-bedrm-count', auto_open=True)

"""###!!The above figure shows that the Accommodates-Price trend line is not linear when #Accommodates is above 10"""

# Property-Type and Price

# fig = px.box(df, x="Property_Type", y="Price", title = 'Price Distribution over Property types', width=1000, height=400)

# fig.show()
# py.plot(fig, filename = 'price-property_type', auto_open=True)

# Number Bedrooms and price

# fig = px.box(df, x="Bedrooms", title = 'Price distributions for each #Bedrooms', y="Price",width=1000, height=400)

# fig.show()
# py.plot(fig, filename = 'bedroom-price', auto_open=True)

# Room type and Price


# plt.figure(figsize=(18, 6))
# chart2 = sns.boxplot(x = df["Room_Type"], y = df["Price"])#, palette="Blues");
# chart2.set_xticklabels(chart2.get_xticklabels(), rotation=0);
# plt.xlabel("Room_Type", size=20)
# plt.ylabel("Price", size=20)
# plt.show()



"""##Airbnb San Francisco Mapping"""

# Mapping based on #Bedrooms

# fig = px.scatter_mapbox(df,  title = "San Francisco Map", lat="Latitude", lon="Longitude", color=df['Neighbourhood'],  width=1000, height=700, zoom=10)
# fig.update_layout(mapbox_style="open-street-map")
# fig.show()


# fig.write_html("map3.html")
# py.plot(fig, filename = 'map3', auto_open=True)

# Define X & y

X_train = df[features].drop(columns='Price')
y_train = df['Price']

# Split into train & test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

X_train.shape, y_train.shape,  X_test.shape, y_test.shape

"""#1 - Using Linear Regression

###1-1 Baseline
"""

y_train.describe()

y_test.describe()

"""###The baseline shows :  
average price  = $199,  
with st_deviation of 138.  
This shows that the price distribution has a big spred
"""


import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# Must put Standard Scaler . later on I need to take out some features based on coef_

linreg = make_pipeline(
    ce.OrdinalEncoder(), 
    SimpleImputer(strategy='mean'), 
    # StandardScaler(), 
    LinearRegression()
)

# Fit the model

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Test
# mean_absolute_error(y_test, y_pred_lr) == mean_absolute_error(y_test, linreg.predict(X_test))



import plotly.graph_objects as go

# fig.update_layout(xaxis={'categoryorder':'category ascending'},  title = 'Linear Regression Coefficients')

# linreg.named_steps['linearregression']

"""#2 - Using RandomForestRegressor"""

from sklearn.metrics import mean_absolute_error

# rfr_important = rfr1.named_steps['randomforestregressor'].n_features_ # to count the #features

# !pip install -U plotly

# Convert Series into Dataframes


# dfi

import plotly.graph_objects as go

# py.plot(fig, filename = 'feature_imp2', auto_open=True)

# rfr1.named_steps['randomforestregressor'].estimators_

# rfr1.named_steps['randomforestregressor'].base_estimator_

# Visualisation y_pred & y_test


"""Let's see the price error distribution.  
We see that the most error is on the lowest price.
"""

# Error calculation:
# drfr['error']

# Error distribution plot

"""#RandomForest Regressor to find the best parameters:"""

from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import make_pipeline
import category_encoders as ce
from category_encoders import OneHotEncoder, OrdinalEncoder
"""#Try again RandomForestRegressor with another method, increase n_iter"""

from scipy.stats import randint, uniform
# pd.DataFrame(search2.cv_results_).sort_values(by='rank_test_score').T


# rfr1.named_steps['randomforestregressor'].estimators_

"""#3a - Using XGBRegressor with pipeline"""

import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

from matplotlib import pyplot
from xgboost import plot_importance


xgbreg1 = make_pipeline(
    ce.OrdinalEncoder(),
    XGBRegressor(n_estimators=10, random_state=42, n_jobs=2, max_depth=4,  learning_rate=0.1)
)

xgbreg1.fit(X_train, y_train)

# Make predictions on the test set
from sklearn.metrics import r2_score

y_pred_xg1 = xgbreg1.predict(X_test)

mse_xg1 = np.round(mean_squared_error(y_test, y_pred_xg1),3)
rmse_xg1 = np.round(np.sqrt(mse_xg1),3)
mae_xg1 = np.round(mean_absolute_error(y_test, y_pred_xg1),3)
r2_xg1 = np.round(r2_score(y_test, y_pred_xg1),3)

print('MAE   :', mae_xg1)
print('MSE   :', mse_xg1)
print('RMSE  :', rmse_xg1)
print('R^2 = :', r2_xg1)



"""#3c. Using XGBoostRegressor without pipeline"""

xg2 = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 20, alpha = 10, n_estimators = 70)

xg2.fit(X_train_encoded,y_train)

y_pred_xg2 = xg2.predict(X_test_encoded)

mse_xg2 = np.round(mean_squared_error(y_test, y_pred_xg2),3)
rmse_xg2 = np.round(np.sqrt(mse_xg2),3)
mae_xg2 = np.round(mean_absolute_error(y_test, y_pred_xg2),3)
r2_xg2 =  np.round(r2_score(y_test, y_pred_xg2),3)

print('MAE   :', mae_xg2)
print('MSE   :', mse_xg2)
print('RMSE  :', rmse_xg2)
print('R^2 = :', r2_xg2)



"""##Visualize Boosting Trees and Feature Importance with XGB"""


# params

# means

# Merics

# Compare with the TRAINING metric

# print('Training MAE:', mean_absolute_error(y_train, search4.predict(X_train_xg4)))

# XGB feature_importances_
# https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/

# PLOTTING, can be done in many ways:

# plot_importance(xgb4, max_num_features=15, height=0.5)#, xlim=(0,10), ylim=(0,10))
# # fig.set_size_inches(4,2);
# pyplot.show()

# or


# or 

# ax4 = xgb.plot_importance(xgb4, max_num_features=15)
# fig = ax4.figure
# fig.set_size_inches(7,5);

# search4.feature_importances_

"""#Which model is the best?  
The best ML model in this dataset is: lowest MAE and biggest R2
"""

# Create dataframe out of all metrics

# Model = ['LinearRegression', 'RFRegression_1', 'RFRegression_2', 'RFRegression_3', 'XGB_1',  'XGB_2' , 'XGB_3']
# MAE = [mae_lr, mae_rfr1, mae_rfr2, mae_rfr3, mae_xg1, mae_xg2, mae_xg4]
# MSE = [mse_lr, mse_rfr1, mse_rfr2, mse_rfr3, mse_xg1, mse_xg2, mse_xg4]
# RMSE = [rmse_lr, rmse_rfr1, rmse_rfr2, rmse_rfr3, rmse_xg1, rmse_xg2, rmse_xg4]
# R2 = [r2_lr, r2_rfr1, r2_rfr2, r2_rfr3, r2_xg1, r2_xg2, r2_xg4]
# Parameter = ['static', 'static', 'static', 'optimized', 'static', 'static', 'opimized' ]

# cols = {'Model':Model, 'MAE':MAE ,  'MSE': MSE,  'RMSE': RMSE, 'R2': R2, 'Parameter': Parameter}

# metric1 = pd.DataFrame(cols)
# # metric1 = pd.DataFrame(cols, index = ['LinearRegression', 'RFRegression', 'RFRegression_best', 'XGB'])

# metric1

# Best_model = metric1.tail(1)
# Best_model



"""#PDP-Plot"""

