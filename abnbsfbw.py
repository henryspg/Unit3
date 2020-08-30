
import pandas as pd
import numpy as np

abnb = pd.read_csv('AbnbSF_clean.csv')
print(abnb.shape)

abnb.info()

print("\nNULL :\n", abnb.isnull().sum())


def wrangle(X):
    """Wrangle train, validate, and test sets in the same way"""
    
    # Prevent SettingWithCopyWarning
    X = X.copy()

    # Fillna for other columns with most frquent values. Definitely with 1 (one)
    cols3 = ['Bedrooms', 'Beds', 'Bathrooms']
    for col in cols3:
        X[col] = X[col].fillna(X[col].mode()[0])

    # return the wrangled dataframe
    return X

abnb = wrangle(abnb)


df = abnb.copy()
df.head(3)
print("\ndescribe price: \n", df['Price'].describe())

# Limit the price: remove 0.
df =   df[(df['Price'].between(20,800))]# 

# Get a dataframe with all train columns except unneeded columns
df_features = df.drop(columns=['Price'], axis=1) ## Without Lat/Long was done on 07/28

# Get a list of the numeric features
numeric_features = df_features.select_dtypes(include='number').columns.tolist()


# Cardinality of the nonnumeric features
cardinality = df_features.select_dtypes(exclude='number').nunique()

# List of all categorical features with  (2< cardinality <70)
cat_features = cardinality[(cardinality <= 70) & (cardinality > 2) ].index.tolist()

features = numeric_features + cat_features
print(features)

train = df[features]

print(train.head(2))

# Define X & y
X_train = df[features]
y_train = df['Price']

# Split into train & test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print(X_train.shape, y_train.shape,  X_test.shape, y_test.shape)


# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, uniform

import xgboost as xgb
from xgboost import XGBRegressor


# Shapley Value & XGBoost

feature_sh = ['Accommodates',  'Bathrooms', 'Bedrooms', 'Beds', 'Minimum_Nights', 'Maximum_Nights', 'Guests_Included'] 

# Define X & y
X_train = df[feature_sh]
y_train = df['Price']


# Split into train & test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X_train.shape, y_train.shape,  X_test.shape, y_test.shape

xgbreg2 = XGBRegressor(n_estimators=100, 
                       random_state=42, 
                       n_jobs=2, 
                       max_depth=4,  
                       learning_rate=0.1)

xgbreg2.fit(X_train, y_train)


import shap 

# Using the predict function
def predict1(Accommodates,	Bedrooms,	Bathrooms,	Beds,	Minimum_Nights,	Maximum_Nights,	Guests_Included):

    # Make dataframe from the inputs
    dshap = pd.DataFrame(
        data=[[Accommodates,	Bathrooms,	Bedrooms,	Beds,	Minimum_Nights,	Maximum_Nights,	Guests_Included]], 
        columns=['Accommodates',	'Bathrooms',	'Bedrooms',	'Beds',	'Minimum_Nights',	'Maximum_Nights',	'Guests_Included']
    )

    # Get the model's prediction
    pred = xgbreg2.predict(dshap)[0]

    result = f'is ${pred:,.0f} \n'
    print(result)

    return pred




Accommodates = 2
Bathrooms= 2
Bedrooms = 2
Beds= 3
Minimum_Nights = 1
Maximum_Nights = 3
Guests_Included = 5



print("\nThe airbnb rent prediction for below features is:")
print("Accommodates =", Accommodates)
print("Bathrooms =", Bathrooms)
print("Bedrooms =", Bedrooms)
print("Beds =", Beds)
print("Minimum_Nights =", Minimum_Nights)
print("Maximum_Nights =", Maximum_Nights)
print("Guests_Included =", Guests_Included)

pred = predict1(Accommodates, Bathrooms, Bedrooms, Beds, Minimum_Nights , Maximum_Nights, Guests_Included)
