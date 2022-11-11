# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 23:03:14 2022

@author: User
"""
from urllib.request import urlopen
import json
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import grangercausalitytests
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly
import plotly.express as px  # (version 4.7.0 or higher)
from dash import dcc, html, Input, Output  # pip install dash (version 2.0.0 or higher)
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix as cm, accuracy_score
from sklearn import preprocessing
import math
import random 
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import grangercausalitytests

def district_analysis(district, food, use_admissions):
    """Takes the name of the region for the analysis and the food of interest,
    accepts 'maize','milk','oil' and 'rice'. use_admissions == True 
    if SAM admission data is analyzed instead of general prevalence"""
      
    #Reading in relevant datasets and merging them
    #If the next row is uncommented you can use the faulty first dataset, which ended up on the poster
    #district_regions = pd.read_csv(your_datapath + 'districts_regions.csv')
    district_regions = pd.read_csv('districts_regions_updated.csv')
    
    correction={'Bulo Burto':"Hiraan","Gebiley":"Woqooyi Galbeed",
                "Hargeysa":"Woqooyi Galbeed","Berbera":"Woqooyi Galbeed"}
    
    support=district_regions.set_index("district").to_dict()["Region"]
    
    def correct(x):
        try:
            return correction[x]
        except KeyError:
            return support[x]
    
    district_regions["Region"]= district_regions["district"].apply(correct)

    prevalence_df = pd.read_csv('prevalence_estimates.csv', parse_dates=['date'])
    prevalence_df = prevalence_df[['date','district','GAM','SAM']]
    prevalence_df = prevalence_df.merge(district_regions)

    admissions = pd.read_csv('admissions.csv')
    admissions = admissions[['date','district','SAM_admissions']]
    admissions = admissions.merge(district_regions)
    admissions = admissions.dropna()

    food_prices = pd.read_csv('food_prices_districts.csv', parse_dates=['date'])
    food_prices = food_prices.drop(['Unnamed: 0'], axis=1)
    food_prices = food_prices[(food_prices['date'] < '2022-01-01')]
    food_prices_filtered = food_prices[['Region','Product','date','Open','Close','High','Low']]

    #Making subsets of the food prices for the product we are interested in
    food_of_interest_price = food_prices_filtered.loc[food_prices_filtered['Product'] == food]

    #Determining in which region a district is located
    region = district_regions.loc[district_regions['district'] == district].iloc[0,0]

    #Making subsets for the districts we are interested in
    food_of_interest_price_region = food_of_interest_price.loc[food_of_interest_price['Region'] == region]
    prevalence_df_district = prevalence_df.loc[prevalence_df['district'] == district]
    prevalence_df_district = prevalence_df_district.sort_values(by = 'date')

    admissions_district = admissions.loc[admissions['district'] == district]
    admissions_district = admissions_district.sort_values(by = 'date')

    if len(prevalence_df_district) < 7 and use_admissions == False:
        #return(district + ': Too few general prevalence datapoints available')
        return {district:"not_enough_data"}

    if len(admissions_district) < 20 and use_admissions:
        #return(district + ': Too few admissions datapoints available')
        return {district:"not_enough_data"}

    #Grouping the datapoints for food price (if there are multiple sampled markets)
    food_prices_avg = food_of_interest_price_region[["date","Open"]].groupby(["date"]).mean()

    #Making the data as stationary by taking the difference
    food_prices_avg['Price change'] = food_prices_avg[['Open']].diff()
    prevalence_df_district['Change'] = prevalence_df_district[['GAM']].diff()
    admissions_district['Change'] = admissions_district[['SAM_admissions']].diff()

    #granger causality
    if use_admissions == False:
        change_list = prevalence_df_district['Change'].to_list()
    if use_admissions:
        change_list = admissions_district['Change'].to_list()

    price_change_list = food_prices_avg['Price change'].to_list()
    granger_df = pd.DataFrame(columns = ['Change', 'Price change'], data = zip(change_list, price_change_list))
    granger_df = granger_df[["Change","Price change"]][1:]
    
    if granger_df.empty:
        return {district:"missing"}
 
    #Convert to string and take the p-value of the ssr F test
    #ssr F tests should be more robust than chi-squared for small sample sizes
    granger_test = str(grangercausalitytests(granger_df[['Change', 'Price change']], 
                                         maxlag = 1,verbose=False)[1][0]["ssr_ftest"][1])
    return {district:granger_test}


def food_casual(district_list, food_type, use_admissions=False):
    final={}
  
    temp={}
 
    for name in district_list:
        output=district_analysis(name, food_type, use_admissions)
        try:
            result= float(output[name])
            if result<0.05:
               
                temp[name]=result
               
        except ValueError:
            pass
    final[food_type]=temp
    return pd.DataFrame(final)  

prevalence_df = pd.read_csv('prevalence_estimates.csv', parse_dates=['date'])

# reading consumer price index data
cpi_df = pd.read_csv('ObservationData_rkvwkxe.csv', parse_dates=['Date', 'base-per'])
cpi_df['date'] = pd.to_datetime(cpi_df['Date'], format='%YM%m')
cpi_weights_df = cpi_df[cpi_df['base-per'] == 'Not Applicable']
cpi_df = cpi_df[cpi_df['base-per'] != 'Not Applicable']
cpi_df['base_date'] = pd.to_datetime(cpi_df['base-per'], format='%YM%m')
cpi_df.drop(columns=['Date', 'base-per'], inplace=True)
cpi_weights_df.drop(columns=['Date'], inplace=True)

cpi_df_pivoted = cpi_df.pivot(index="date", columns="indicator", values=["Value", "base_date"]).reset_index()
cpi_df_pivoted['base date'] = cpi_df_pivoted['base_date']['TRANSPORT']
cpi_df_pivoted.drop(columns='base_date', inplace=True)
cpi_df_pivoted.columns = cpi_df_pivoted.columns.get_level_values(1)
cpi_df_pivoted.columns = ['date', 'ALCOHOLIC BEVERAGES, TOBACCO AND NARCOTICS', 'ALL GROUPS',
       'CLOTHING AND FOOTWEAR', 'COMMUNICATION', 'EDUCATION',
       'FOOD AND NON ALCOHOLIC BEVERAGES',
       'FURNISHINGS, HOUSEHOLD EQUIPMENT AND ROUTINE MAINTENANCE OF THE HOUSE',
       'HEALTH', 'HOUSING, WATER, ELECTRICITY, GAS AND OTHER FUELS',
       'MISCELLANEOUS GOODS AND SERVICES', 'RECREATION AND CULTURE',
       'RESTAURANTS AND HOTELS', 'TRANSPORT', 'base date']

cols = cpi_df_pivoted.columns[cpi_df_pivoted.dtypes.eq('object')]
cpi_df_pivoted[cols] = cpi_df_pivoted[cols].apply(pd.to_numeric, errors='coerce')
cpi_df_grouped = cpi_df_pivoted.groupby(pd.Grouper(key='date', freq='6M')).mean()
cpi_df_grouped = cpi_df_grouped.reset_index()
cpi_df_grouped['date'] = cpi_df_grouped['date'].apply(lambda x : x.replace(day=1))


districts = prevalence_df['district'].unique()
df_list = []
for district in districts:
    food_df = pd.read_csv('food_prices_districts.csv', parse_dates=['date'])
    food_df.drop(['Unnamed: 0', 'Country', 'Currency', "Region"], axis=1, inplace=True)

    prods_df = pd.get_dummies(food_df['Product'])
    prods_df['maize'] = food_df['High'] * prods_df['maize'] 
    prods_df['oil'] = food_df['High'] * prods_df['oil'] 
    prods_df['milk'] = food_df['High'] * prods_df['milk'] 
    prods_df['rice'] = food_df['High'] * prods_df['rice'] 
    prods_df['date'] = food_df['date']
    prods_df['district'] = food_df['district']

    food_df = prods_df.groupby([pd.Grouper(key='date' , freq='6M'), 'district']).mean()
    food_df = food_df.reset_index()
    food_df['date'] = food_df['date'].apply(lambda x : x.replace(day=1))
    
    prevalence_df = pd.read_csv('prevalence_estimates.csv', parse_dates=['date'])

    covid_df = pd.read_csv('covid.csv', parse_dates=['date'])
    ipc_df = pd.read_csv('ipc2.csv', parse_dates=['date'])
    risk_df = pd.read_csv('FSNAU_riskfactors.csv', parse_dates=['date'])
    production_df = pd.read_csv('production.csv', parse_dates=['date'])
    admissions_df = pd.read_csv('admissions.csv', parse_dates=['date'])
    conflict_df = pd.read_csv('conflict.csv', parse_dates=['date'])
    locations_df = pd.read_csv('locations.csv', parse_dates=['date'])

    #Select data for specific district
    prevalence_df = prevalence_df[prevalence_df['district']==district]
    ipc_df = ipc_df[ipc_df['district']==district]
    risk_df = risk_df[risk_df['district']==district]
    production_df = production_df[production_df['district']==district] 
    admissions_df = admissions_df[admissions_df['district']==district]
    conflict_df = conflict_df[conflict_df['district']==district]
    locations_df = locations_df[locations_df['district']==district]
    food_df = food_df[food_df['district']==district]
    
    
    risk_df = risk_df.groupby(pd.Grouper(key='date', freq='6M')).mean()
    risk_df = risk_df.reset_index()
    risk_df['date'] = risk_df['date'].apply(lambda x : x.replace(day=1))
    
    covid_df = covid_df.groupby(pd.Grouper(key='date', freq='6M')).sum()
    covid_df = covid_df.reset_index()
    covid_df['date'] = covid_df['date'].apply(lambda x : x.replace(day=1))
    
    production_df['cropdiv'] = production_df.count(axis=1)
    
    admissions_df = admissions_df.groupby(pd.Grouper(key='date', freq='6M')).mean()
    admissions_df = admissions_df.reset_index()
    admissions_df['date'] = admissions_df['date'].apply(lambda x : x.replace(day=1))
    
    conflict_df = conflict_df.groupby(pd.Grouper(key='date', freq='6M')).mean()
    conflict_df = conflict_df.reset_index()
    conflict_df['date'] = conflict_df['date'].apply(lambda x : x.replace(day=1))
    
    ipc_df = ipc_df.groupby(pd.Grouper(key='date', freq='6M')).mean()
    ipc_df = ipc_df.reset_index()
    ipc_df['date'] = ipc_df['date'].apply(lambda x : x.replace(day=1))
    
    locations_df = locations_df.groupby(pd.Grouper(key='date', freq='6M')).mean()
    locations_df = locations_df.reset_index()
    locations_df['date'] = locations_df['date'].apply(lambda x : x.replace(day=1))
    
    #Sort dataframes on date
    
    prevalence_df.sort_values('date', inplace=True)
    covid_df.sort_values('date', inplace=True)
    ipc_df.sort_values('date', inplace=True)
    risk_df.sort_values('date', inplace=True)
    production_df.sort_values('date', inplace=True)
    
    admissions_df.sort_values('date', inplace=True)
    conflict_df.sort_values('date', inplace=True)
    ipc_df.sort_values('date', inplace=True)
    locations_df.sort_values('date', inplace=True)
    food_df.sort_values('date', inplace=True)
    
    #Merge dataframes, only joining on current or previous dates as to prevent data leakage
    df = pd.merge_asof(left=prevalence_df, right=risk_df, direction='backward', on='date')
    #df = pd.merge_asof(left=df, right=ipc_df, direction='backward', on='date')
    #df = pd.merge_asof(left=df, right=production_df, direction='backward', on='date')
    #df = pd.merge_asof(left=df, right=risk_df, direction='backward', on='date')
    #df = pd.merge_asof(left=df, right=covid_df, direction='backward', on='date')
    df = pd.merge_asof(left=df, right=admissions_df, direction='backward', on='date')
    #df = pd.merge_asof(left=df, right=conflict_df, direction='backward', on='date')
    df = pd.merge_asof(left=df, right=locations_df, direction='backward', on='date')
    df = pd.merge_asof(left=df, right=food_df, direction='backward', on='date')
    
    df['prevalence_6lag'] = df['GAM Prevalence'].shift(1)
    df['next_prevalence'] = df['GAM Prevalence'].shift(-1)
    
    df = df.rename(columns={"GAM Prevalence": "prevalence"})
    
    df['month'] = df['date'].dt.month
    
    #Add target variable: increase for next month prevalence (boolean)
    increase = [False if x[1]<x[0] else True for x in list(zip(df['prevalence'], df['prevalence'][1:]))]
    increase.append(False)
    df['increase'] = increase
    df.iloc[-1, df.columns.get_loc('increase')] = np.nan #No info on next month
    
    #Add target variable: increase for next month prevalence (boolean)
    increase_numeric = [x[1] - x[0] for x in list(zip(df['prevalence'], df['prevalence'][1:]))]
    increase_numeric.append(0)
    df['increase_numeric'] = increase_numeric
    df.iloc[-1, df.columns.get_loc('increase_numeric')] = np.nan #No info on next month
    
    df.loc[(df.date < pd.to_datetime('2020-03-01')), 'covid'] = 0
    df['district'] = district
    df_list.append(df)
    
df = pd.concat(df_list, ignore_index=True)
df['district_encoded'] = df['district'].astype('category').cat.codes

df.fillna(df.median(), inplace=True)            

df.sort_values('date', inplace=True)
df.reset_index(inplace=True, drop=True)

dates = df['date'].unique()

df_train = df[(df['date'] != dates[-1]) & (df['date'] != dates[-2])]
df_test = df[(df['date'] == dates[-1]) | (df['date'] == dates[-2])]

Xtrain = df_train.drop(columns = ['Unnamed: 0','increase_numeric', 
                                'next_prevalence', 'Maize prices', 
                                ])
ytrain = df_train['next_prevalence'] .values  

Xtest = df_test.drop(columns = ['Unnamed: 0','increase_numeric', 
                                'next_prevalence', 'Maize prices', 
                                ])
ytest = df_test['next_prevalence'] .values  


Xtrain = Xtrain._get_numeric_data()
Xtest = Xtest._get_numeric_data()

Xtrain = Xtrain.loc[:, (Xtrain != 0).any(axis=0)]
Xtest = Xtest.loc[:, (Xtest != 0).any(axis=0)]
parameter_scores = []

clf = RandomForestRegressor(random_state=0)
clf.fit(Xtrain, ytrain)
predictions = clf.predict(Xtest)

df_test['predictions'] = predictions
df_test['errors'] = df_test['next_prevalence'] - df_test['predictions']
rez = df_test[['date', 'district', 'next_prevalence', 'predictions', 'errors']]

MAE = mean_absolute_error(ytest, predictions)

#Generate boolean values for increase or decrease in prevalence. 0 if next prevalence is smaller than current prevalence, 1 otherwise.
increase           = [0 if x<y else 1 for x in df_test['next_prevalence'] for y in df_test['prevalence']]
predicted_increase = [0 if x<y else 1 for x in predictions                for y in df_test['prevalence']]

#Calculate accuracy of predicted boolean increase/decrease
acc = accuracy_score(increase, predicted_increase)

#Print model scores
print('-----------------------')
print('Model scores:')
print("Mean Absolute Error RF: ", MAE, ", accuracy RF: ", acc)
print('-----------------------')


print("Causality:")
district_list = ['Adan Yabaal', 'Caluula', 'Jowhar', 'Kurtunwaarey', 'Luuq', 'Qandala']
print(food_casual(district_list,"maize",False))
print(food_casual(['Iskushuban'],"maize",True))
print(food_casual(['Ceel Waaq', 'Jamaame', 'Qansax Dheere'],"rice",False))
print(food_casual(['Jariiban'],"rice",True))
print('-----------------------')

def plot_feature_importances(model, columns):
    feat_importances = pd.Series(model.feature_importances_, index=columns)
    feat_importances.nlargest(9).plot(kind='barh')
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.show()
    
#plot_feature_importances(clf, Xtrain.columns)

rez.to_csv('rezult.csv')

# Open geojson
with urlopen(
        'https://data.humdata.org/dataset/dc9d0eaa-9664-463b-9b67-0b440eff662a/resource/d4949fb4-d4b0-481c-ab7c-ca7f1e3550c0/download/somalia_districts.geojson') as response:
    counties_s = json.load(response)


# Open output file of baseline with predictions
df_pred = pd.read_csv('rezult.csv')
# Open output file of baseline
df_base = pd.read_csv('baseline_output.csv')


# Replace district names to math geojson districts
attribute = 'district'
replace_dict = {'Saakow/Salagle': 'Saakow', 'Wanla Weyn': 'Wanle Weyne', 'Banadir': 'Mogadishu', 'Gebiley': 'Gabiley',
                'Garoowe': 'Garowe', 'Gaalkacyo': 'Galkaacyo', 'Baydhaba/Bardaale': 'Baidoa',
                'Ceel Waaq': 'El Waq', 'Bulo Burto': 'Bulo Burti', 'Laasqoray/Badhan': 'Laasqoray',
                'Kismaayo': 'Kismayo', 'Bandarbeyla': 'Bandarbayla', 'Afmadow/Xagar': 'Afmadow'}
df_base.replace({attribute: replace_dict}, inplace=True)
df_pred.replace({attribute: replace_dict}, inplace=True)

# ## Set up causality


# Reading in relevant datasets and merging them
district_regions = pd.read_csv('districts_regions.csv')  # get it from the google drive ,in the map additional data
district_list = district_regions['district'].to_list()
prevalence_df = pd.read_csv('prevalence_estimates.csv', parse_dates=['date'])
prevalence_df = prevalence_df[['date', 'district', 'GAM', 'SAM']]
prevalence_df = prevalence_df.merge(district_regions)
food_prices = pd.read_csv('food_prices_districts.csv', parse_dates=['date'])  # get is from the drive
food_prices = food_prices.drop(['Unnamed: 0'], axis=1)
food_prices = food_prices[(food_prices['date'] < '2022-01-01')]
food_prices_filtered = food_prices[['Region', 'Product', 'date', 'Open', 'Close', 'High', 'Low']]


def district_analysis(district, food):
    """Takes the name of the region for the analysis and the food of interest,
    accepts 'maize','milk','oil' and 'rice' """

    # TO INCLUDE: SAM ADMISSIONS AS OPTION, MISSING DISTRICTS
    # TO DO: WHERE ARE THE SIGNIFICANT DISTRICTS LOCATED? IS IT DIFFERENT IF WE USE RICE???

    # Making subsets of the food prices for the product we are interested in
    food_of_interest_price = food_prices_filtered.loc[food_prices_filtered['Product'] == food]

    # Determining in which region a district is located
    region = district_regions.loc[district_regions['district'] == district].iloc[0, 0]

    # Making subsets for the districts we are interested in
    food_of_interest_price_region = food_of_interest_price.loc[food_of_interest_price['Region'] == region]
    print("food_of_interest_price_region")
    prevalence_df_district = prevalence_df.loc[prevalence_df['district'] == district]
    prevalence_df_district = prevalence_df_district.sort_values(by='date')

    if len(prevalence_df_district) < 7:
        return {district: "Too few datapoints available"}

    # Grouping the datapoints for food price (if there are multiple sampled markets)
    food_prices_avg = food_of_interest_price_region[["date", "Open"]].groupby(["date"]).mean()

    # Making the data as stationary as the data allows
    food_prices_avg['Price change'] = food_prices_avg[['Open']].diff()
    prevalence_df_district['Change'] = prevalence_df_district[['GAM']].diff()

    # granger causality
    change_list = prevalence_df_district['Change'].to_list()
    price_change_list = food_prices_avg['Price change'].to_list()
    granger_df = pd.DataFrame(columns=['Change', 'Price change'], data=zip(change_list, price_change_list))
    # print( granger_df)

    granger_df = granger_df[["Change", "Price change"]][1:]
    if granger_df.empty:
        return {district: "Too few datapoints available"}

    # Maxlag = 2 gave clutters the screen, might make it a parameter though
    # Convert to string and take the p-value of the ssr F test
    granger_test = str(grangercausalitytests(granger_df[['Change', 'Price change']],
                                             maxlag=1)[1][0]["ssr_ftest"][1])

    return {district: granger_test}


food_list = ['maize', 'milk', 'oil', 'rice']


def food_csl(lst_food):
    food_dict = {}  # key:food type , value which districts are statistcally signifact
    results = []
    for food in lst_food:
        for i in district_list:  # len(district_list)
            granger = district_analysis(i, food)
            output = granger[i]
            if output != "Too few datapoints available":
                output = float(output)
                if output < 0.05:
                    results.append(i)
        food_dict[food] = results
    return food_dict


# ## Dashboard
# ### Set up causality

def plot_sig(district, food):
    """Takes the name of the region for the analysis and the food of interest,
    accepts 'maize','milk','oil' and 'rice' """

    # TO INCLUDE: SAM ADMISSIONS AS OPTION, MISSING DISTRICTS
    # TO DO: WHERE ARE THE SIGNIFICANT DISTRICTS LOCATED? IS IT DIFFERENT IF WE USE RICE???

    # Reading in relevant datasets and merging them
    district_regions = pd.read_csv('districts_regions.csv')
    prevalence_df = pd.read_csv('prevalence_estimates.csv', parse_dates=['date'])
    prevalence_df = prevalence_df[['date', 'district', 'GAM', 'SAM']]
    prevalence_df = prevalence_df.merge(district_regions)
    food_prices = pd.read_csv('food_prices_districts.csv', parse_dates=['date'])
    food_prices = food_prices.drop(['Unnamed: 0'], axis=1)
    food_prices = food_prices[(food_prices['date'] < '2022-01-01')]
    food_prices_filtered = food_prices[['Region', 'Product', 'date', 'Open', 'Close', 'High', 'Low']]

    # Making subsets of the food prices for the product we are interested in
    food_of_interest_price = food_prices_filtered.loc[food_prices_filtered['Product'] == food]

    # Determining in which region a district is located
    region = district_regions.loc[district_regions['district'] == district].iloc[0, 0]

    # Making subsets for the districts we are interested in
    food_of_interest_price_region = food_of_interest_price.loc[food_of_interest_price['Region'] == region]

    prevalence_df_district = prevalence_df.loc[prevalence_df['district'] == district]
    prevalence_df_district = prevalence_df_district.sort_values(by='date')

    # Grouping the datapoints for food price (if there are multiple sampled markets)
    food_prices_avg = food_of_interest_price_region[["date", "Open"]].groupby(["date"]).mean()

    # Making the data as stationary as the data allows
    food_prices_avg['Price change'] = food_prices_avg[['Open']].diff()
    prevalence_df_district['Change'] = prevalence_df_district[['GAM']].diff()

    # granger causality
    change_list = prevalence_df_district['Change'].to_list()
    price_change_list = food_prices_avg['Price change'].to_list()
    granger_df = pd.DataFrame(columns=['Change', 'Price change'], data=zip(change_list, price_change_list))
    # print( granger_df)

    granger_df = granger_df[["Change", "Price change"]][1:]
    return granger_df


# ## Causality plot


# Play around with these lists
district = district_list[:2]
food = food_list[:2]

# Check if district and food are actualy lists and do not contain one individual element
if type(district) is not list:
    district = [district]
if type(food) is not list:
    food = [food]

max_row = len(district)
max_col = len(food)

cols = plotly.colors.DEFAULT_PLOTLY_COLORS

causality_graph = make_subplots(rows=max_row, cols=max_col, shared_xaxes=True, shared_yaxes=True)

i_index = 1
for i in district:
    j_index = 1
    for j in food:
        plot = plot_sig(i, j)
        plot['index'] = plot.index
        if j_index == 1 and i_index == 1:
            iflegend = True
        else:
            iflegend = False
        causality_graph.add_trace(go.Scatter(x=plot['index'], y=plot['Change'],
                                             name='Change', line=dict(width=2, color=cols[0]), showlegend=iflegend),
                                  row=i_index, col=j_index)
        causality_graph.add_trace(go.Scatter(x=plot['index'], y=plot['Price change'],
                                             name='Price change', line=dict(width=2, color=cols[1]),
                                             showlegend=iflegend),
                                  row=i_index, col=j_index)
        causality_graph.update_yaxes(title_text=i, row=i_index, col=1)
        if i_index == max_row:
            causality_graph.update_xaxes(title_text=j, row=i_index, col=j_index)
        j_index += 1
    i_index += 1
    causality_graph.update_layout(title_text='Causality comparison', height=(200 * max_row))

# ### Components


# Text
title = html.H1('Dashboard Somalia Malnutrition Predictions', style={'textAlign': 'center', 'color': '#1F1F1F'})
subtitle = html.I('Data challenge 3, Group 6, 2022-2023', style={'color': '#1F1F1F'})

titlebox = html.Div([title, subtitle])  # , style={'background' : '#F0D1AD'}


# Tabs
tabs = dcc.Tabs(id="tabs", value='tab_1', children=[
    dcc.Tab(label='Map', value='tab_1'),
    dcc.Tab(label='Causality', value='tab_2')])
tabs_content = html.Div(id='tabs_content')

# ### Layout and App updates

# Global Layout
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    titlebox,
    html.Br(),
    tabs,
    tabs_content
])


# ------------------------------------------------------------------------------
# App updates

# Tab content
@app.callback(Output('tabs_content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab_1':
        return html.Div([
            html.H3('Somalia map'),
            dcc.Dropdown(id='pred_drop',
                         options={
                             'pred': 'Predictions',
                             'error': 'Errors',
                             'nxt': 'True values'},
                         value='pred',
                         multi=False),
            html.Br(),
            dcc.Dropdown(id='date_drop',
                         options={
                             'jan': 'January 2021',
                             'jul': 'July 2021'},
                         value='jan',
                         multi=False),
            html.Br(),
            html.Div(id='output_container', children=[]),
            html.Br(),
            dcc.Graph(id='baseline_map', figure={}, style={'width': '50%', 'display': 'inline-block'}),
            dcc.Graph(id='somalia_map', figure={}, style={'width': '50%', 'display': 'inline-block'})
        ])
    elif tab == 'tab_2':
        return html.Div([
            html.H3('Causality analysis'),
            html.Div(children = [
                    html.P('Choose districts:'), 
                    dcc.Dropdown(id='district_dropdown', options = district_list, value = district_list[:2], 
                                         searchable=True, clearable=False, multi=True, style = {'width' : '85%'}),
                    html.P('Choose food types:'), 
                    dcc.Dropdown(id='food_dropdown', options = food_list, value = food_list[:2], 
                                 searchable=True, clearable=False, multi=True, style = {'width' : '85%'})
                ], style = {'width' : '50%', 'display' : 'inline-block'}), 
            html.Div([
                    html.P('Enter minimum y value < 0 (in steps of 500):'), 
                    dcc.Input(id='y_min', type='number', max = 0, step=500),
                    html.P('Enter maximum y value > 0 (in steps of 500):'), 
                    dcc.Input(id='y_max', type='number', min = 0, step=500)
                ], style = {'width' : '50%', 'display' : 'inline-block'}), 
            dcc.Graph(id='causality_plot', figure= causality_graph)
        ])


# Connect the Plotly graphs with Dash Components
@app.callback(
    [Output(component_id='output_container', component_property='children'),
     Output(component_id='somalia_map', component_property='figure'),
     Output(component_id='baseline_map', component_property='figure')
     ],
    [Input(component_id='date_drop', component_property='value'),
     Input(component_id='pred_drop', component_property='value')]
)
def update_map(date_slctd, pred_slctd):
    if pred_slctd != 'error':
        min_value = 0
        max_value = 0.6
        if pred_slctd == 'nxt':
            pred_container = 'true values'
            pred_df = 'next_prevalence'
        elif pred_slctd == 'pred':
            pred_container = 'predictions'
            pred_df = pred_container
    else:
        min_value = -0.2
        max_value = 0.2
        pred_container = 'errors'
        pred_df = pred_container

    if date_slctd == 'jan':
        container = f'Visualize {pred_container} of January 2021'
        date = '2021-01-01'
    else:
        container = f'Visualize {pred_container} of July 2021'
        date = '2021-07-01'

    district_data_pred = df_pred[df_pred['date'] == date]
    district_data_base = df_base[df_base['date'] == date]

    # Somalia map
    somalia_map = px.choropleth_mapbox(district_data_pred, geojson=counties_s, featureidkey="properties.DISTRICT",
                                       locations='district', color=pred_df,
                                       color_continuous_scale=[[0.0, "rgb(65,137,221)"],
                                                               [1.0, "rgb(221,149,65)"]],
                                       range_color=(min_value, max_value),
                                       mapbox_style="carto-positron",
                                       zoom=4.5, center={"lat": 5.787273, "lon": 46.439236},
                                       opacity=0.5,
                                       )
    somalia_map.update_layout(title_text='Our model map', height=650)
    # Somalia map
    baseline_map = px.choropleth_mapbox(district_data_base, geojson=counties_s, featureidkey="properties.DISTRICT",
                                        locations='district', color=pred_df,
                                        color_continuous_scale=[[0.0, "rgb(65,137,221)"],
                                                                [1.0, "rgb(221,149,65)"]],
                                        range_color=(min_value, max_value),
                                        mapbox_style="carto-positron",
                                        zoom=4.5, center={"lat": 5.787273, "lon": 46.439236},
                                        opacity=0.5,
                                        )
    baseline_map.update_layout(title_text='Baseline map', height=650)

    return container, somalia_map, baseline_map


# Update causality plots
@app.callback(
    [Output(component_id='causality_plot', component_property='figure')],
    [Input(component_id='district_dropdown', component_property='value'),
    Input(component_id='food_dropdown', component_property='value'),
    Input(component_id='y_min', component_property='value'),
    Input(component_id='y_max', component_property='value')]
)
def update_causality(district_lst, food_lst, y_min, y_max):
    if type(district_lst) is not list:
        district = [district_lst]
    else:
        district = district_lst
    if type(food_lst) is not list:
        food = [food_lst]
    else:
        food = food_lst

    max_row = len(district)
    max_col = len(food)
    
    causality_graph = make_subplots(rows=max_row, cols=max_col, shared_xaxes = True, shared_yaxes=True)

    i_index = 1
    for i in district:
        j_index = 1
        for j in food:
            plot = plot_sig(i,j)
            plot['index'] = plot.index
            if j_index == 1 and i_index == 1:
                iflegend = True
            else:
                iflegend = False
            causality_graph.add_trace(go.Scatter(x = plot['index'], y = plot['Change'], 
                                     name='Prevalance', line=dict(width=2, color="rgb(221,149,65)"), showlegend = iflegend),
                          row = i_index, col = j_index)
            causality_graph.add_trace(go.Scatter(x = plot['index'], y = plot['Price change'], 
                                     name = 'Price change', line=dict(width=2, color="rgb(65,137,221)"), showlegend = iflegend), 
                          row = i_index, col = j_index)
            if y_min is not None and y_max is not None:
                causality_graph.update_yaxes(title_text=i, row=i_index, col=1, range=[y_min, y_max])
            else:
                causality_graph.update_yaxes(title_text=i, row=i_index, col=1)
            if i_index == max_row:
                causality_graph.update_xaxes(title_text=j, row=i_index, col=j_index)
            j_index += 1
        i_index += 1
        
    causality_graph.update_layout(title_text = 'Causality comparison', height = (175*max_row+150))
    
    return [causality_graph]


# ### Run application

if __name__ == "__main__":
    app.run_server(debug=False)