# -*- coding: utf-8 -*-
"""
Created on Tue May 30 15:25:32 2017

@author: Ash
"""

"""
The data scientists at BigMart have collected 2013 sales data for 1559 products
across 10 stores in different cities. Also, certain attributes of each product
and store have been defined. The aim is to build a predictive model and 
find out the sales of each product at a particular store. Using this model, 
BigMart will try to understand the properties of products and stores which
play a key role in increasing sales.

"""

"""
1. The Hypotheses:
    Store Level Hypotheses:
        1. City type: Stores located in urban or Tier 1 cities should have higher sales because of the higher income levels of people there.
        2. Population Density: Stores located in densely populated areas should have higher sales because of more demand.
        3. Store Capacity: Stores which are very big in size should have higher sales as they act like one-stop-shops and people would prefer getting everything from one place
        4. Competitors: Stores having similar establishments nearby should have less sales because of more competition.
        5. Marketing: Stores which have a good marketing division should have higher sales as it will be able to attract customers through the right offers and advertising.
        6. Location: Stores located within popular marketplaces should have higher sales because of better access to customers.
        7. Customer Behavior: Stores keeping the right set of products to meet the local needs of customers will have higher sales.
        8. Ambiance: Stores which are well-maintained and managed by polite and humble people are expected to have higher footfall and thus higher sales.

    Product Level Hypotheses:
        1. Brand: Branded products should have higher sales because of higher trust in the customer.
        2. Packaging: Products with good packaging can attract customers and sell more.
        3. Utility: Daily use products should have a higher tendency to sell as compared to the specific use products.
        4. Display Area: Products which are given bigger shelves in the store are likely to catch attention first and sell more.
        5. Visibility in Store: The location of product in a store will impact sales. Ones which are right at entrance will catch the eye of customer first rather than the ones in back.
        6. Advertising: Better advertising of products in the store will should higher sales in most cases.
        7. Promotional Offers: Products accompanied with attractive offers and discounts will sell more.

Link up between variables and hypothesis:
    
    Variable                  Description                                   Relation to Hyphothesis
    
    item_identifier           Unique Product id                             ID variable 
    Item_Weight               Weight of the product                         Not considered in hypothesis
    Item_Fat_Content          Whether the product is low fat or not         Linked to utility hyphothesis as low
                                                                            low fat items have more sell. 
    Item_Visibility           The % of allocated area of all products       Linked to Display area hypothesis
                              in a store allocated to the particular 
                              product
    Item_Type                 The category to which the product             More inference about utility can be derived 
                              belongs                                       from this 
    Item_MRP                  Maximum retail price (M.R.P)                  Not considered
    Outlet_Identifier         Unique Store ID                               ID variable
    Outlet_establishment_year year of establishment                         Not considered
    Outlet_Size               Ground Area                                   Linked to store capacity hypothesis
    Outlet_loc_type           Type of the city whr store is located         Linked to city type hypothesis
    Outlet_Type               grocery or supermarket                        Linked to store capacity hyphthesis
    Item_Outlet_sales         sales of the product in particular store      Outcome variable
"""


# 2. Data Exploration:
    
    
import pandas as pd 
import numpy as np 


# Read Test and Train data 

train_data = pd.read_csv('Train_data.csv')
print train_data.head()
print train_data.info()

test_data = pd.read_csv('Test_data.csv')
#print(test_data)

"""
Its generally a good idea to combine both train and test data sets into one, 
perform feature engineering and then divide them later again.
"""

"""
 Lets combine them into a dataframe ‘data’ with a ‘source’ 
 column specifying where each observation belongs.
"""

train_data['source'] = 'train'
print(train_data.shape)
test_data['source'] = 'test'
print(test_data.shape)

data = pd.concat([train_data,test_data], ignore_index = True)
print data.head()
print(data.shape)

print(data.apply(lambda x: sum(x.isnull())))

"""
Note that the Item_Outlet_Sales is the target variable and missing values 
are ones in the test set. So we need not worry about it. But we’ll impute
the missing values in Item_Weight and Outlet_Size in the data cleaning 
section.

"""


print(data.describe())
print(data.info())

"""
Some observations:

1. Item_Visibility has a min value of zero. This makes no practical sense because
when a product is being sol=d in a store, the visibility cannot be 0.

2. Outlet_Establishment_Years vary from 1985 to 2009. The values might not be apt
in this form. Rather, if we can convert them to how old the particular store is,
it should have a better impact on sales.
The lower ‘count’ of Item_Weight and Item_Outlet_Sales confirms the findings from the missing value check.
"""

# Moving to nominal (categorical) variable, lets have a look at the number of 
# unique values in each of them.

#print(data.apply(lambda x: len(x.unique())))


#==============================================================================
# This tells that there are 1559 products and 10 outlets/stores 
# (which was also mentioned in problem statement). 
# Another thing that should catch attention is that Item_Type has 16 
# unique values. Let’s explore further using the frequency of different
# categories in each nominal variable. I’ll exclude the ID and 
# source variables for obvious reasons.
# 
#==============================================================================


#Filter categorical variables
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']
#Exclude ID cols and source:
categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier','source']]
print categorical_columns
#Print frequency of categories
#==============================================================================
for col in categorical_columns:
    print ('\nFrequency of Categories for varible %s'%col)
    print (data[col].value_counts())
#==============================================================================
    
#==============================================================================
# Following Observation:
#     
# 1. Item_Fat_Content: Some of ‘Low Fat’ values mis-coded as ‘low fat’ and ‘LF’. Also, some of ‘Regular’ are mentioned as ‘regular’.
# 2. Item_Type: Not all categories have substantial numbers. It looks like combining them can give better results.
# 3. Outlet_Type: Supermarket Type2 and Type3 can be combined. But we should check if that’s a good idea before doing it.
# 
#==============================================================================


# 3. Data Cleaning

"""
This step typically involves imputing missing values and treating 
outliers. Though outlier removal is very important in regression 
techniques, advanced tree based algorithms are impervious to outliers.
So I’ll leave it to you to try it out. We’ll focus on the imputation 
step here, which is a very important step.

"""

# Imputing Missing Values

"""
We found two variables with missing values – Item_Weight and Outlet_Size.
Lets impute the former by the average weight of the particular item.
This can be done as:
    
"""

#Determine the average weight per item:
item_avg_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier')
print item_avg_weight

#Get a boolean variable specifying missing Item_Weight values
miss_bool = data['Item_Weight'].isnull() 
print miss_bool

#Impute data and check #missing values before and after imputation to confirm
#print ('Orignal #missing: %d'% sum(miss_bool))
data.loc[miss_bool,'Item_Weight'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: item_avg_weight[x])
#print ('Final #missing: %d'% sum(data['Item_Weight'].isnull()))

# This confirms that the column has no missing values now.
#  Lets impute Outlet_Size with the mode of the Outlet_Size for the particular type of outlet.

#Import mode function:
from scipy.stats import mode

# Determing the mode for each
print(mode(data['Outlet_Type']))
print(mode(data['Outlet_Type']).mode[0])
data['Outlet_Type'].fillna(mode(data['Outlet_Type']).mode[0], inplace=True)
print(len(data))
print(len(data["Outlet_Type"]))

#Get a boolean variable specifying missing Item_Weight values
miss_bool = data['Outlet_Type'].isnull() 
print(miss_bool.value_counts())

data = data.fillna({"Outlet_Size": "Medium"})
#Get a boolean variable specifying missing Item_Weight values
miss_bool1 = data['Outlet_Size'].isnull() 
print(miss_bool1.value_counts())

#data.to_csv('ddh.csv')

# 4. Feature Engineering

# now see the weight of Outlet type 

print(data.pivot_table(values = 'Item_Outlet_Sales', index = 'Outlet_Type'))
print(data.pivot_table(values = 'Item_Outlet_Sales', index = 'Item_Fat_Content'))

# Modify Item_Visibility

#Determine average visibility of a product
visibility_avg = data.pivot_table(values='Item_Visibility', index='Item_Identifier')
print(visibility_avg)

#Impute 0 values with mean visibility of that product:
miss_bool = (data['Item_Visibility'] == 0)

print ('Number of 0 values initially: %d'%sum(miss_bool))
data.loc[miss_bool,'Item_Visibility'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: visibility_avg[x])
print ('Number of 0 values after modification: %d'%sum(data['Item_Visibility'] == 0))

#Determine another variable with means ratio
data['Item_Visibility_MeanRatio'] = data.apply(lambda x: x['Item_Visibility']/visibility_avg[x['Item_Identifier']], axis=1)
print (data['Item_Visibility_MeanRatio'].describe())


#Get the first two characters of ID:
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])

print(data['Item_Type_Combined'].head())
#Rename them to more intuitive categories:
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
print(data['Item_Type_Combined'].value_counts())

#Years:
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
print(data['Outlet_Years'].head())
print(data['Outlet_Years'].describe())

#data.to_csv('nn.csv')
# Step 5: Modify categories of Item_Fat_Content

# Change categories of low fat:
print ('Original Categories:')
print (data['Item_Fat_Content'].value_counts())

print ('\nModified Categories:')
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})
print (data['Item_Fat_Content'].value_counts())






#Mark non-consumables as separate category in low_fat:
data.loc[data['Item_Type_Combined'] == "Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
data['Item_Fat_Content'].value_counts()

#data.to_csv('nnn.csv')

# Step 6: Numerical and One-Hot Coding of Categorical variables

"""
Since scikit-learn accepts only numerical variables, I converted all categories
of nominal variables into numeric types. Also, I wanted Outlet_Identifier as a
variable as well. So I created a new variable ‘Outlet’ same as Outlet_Identifier
and coded that. Outlet_Identifier should remain as it is, because it will be 
required in the submission file.

Lets start with coding all categorical variables as numeric using ‘LabelEncoder’
from sklearn’s preprocessing module.

"""

#Import library:
    
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

#New variable for outlet
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])
    
    


#One Hot Coding:
data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                              'Item_Type_Combined','Outlet'])




#data.to_csv("transform_dataa.csv")

# Step 7: Exporting Data

"""
Final step is to convert data back into train and test data sets. 
Its generally a good idea to export both of these as modified data sets so that
they can be re-used for multiple sessions. This can be achieved using following code:
    
"""

#Define target and ID columns:
target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']
from sklearn import cross_validation, metrics
def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    #Perform cross-validation:
    cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20, scoring='mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    #Print model report:
    print( "\nModel Report")
    print ("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
    print ("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    
    #Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])
    
    #Export submission file:
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    submission.to_csv(filename, index=False)

#Drop the columns which have been converted to different types:
data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

#Divide into test and train:
train = data.loc[data['source']=="train"]
test = data.loc[data['source']=="test"]

#Drop unnecessary columns:
test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)

#Export files as modified versions:
#train.to_csv("train_modified.csv",index=False)
#test.to_csv("test_modified.csv",index=False)
#

# Model Building 
# Decision Tree


#==============================================================================
from sklearn.tree import DecisionTreeRegressor
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg3 = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
modelfit(alg3, train, test, predictors, target, IDcol, 'alg3.csv')
coef3 = pd.Series(alg3.feature_importances_, predictors).sort_values(ascending=False)
coef3.plot(kind='bar', title='Feature Importances')
#==============================================================================

# Linear regression 

from sklearn.linear_model import LinearRegression, Ridge, Lasso
predictors = [x for x in train.columns if x not in [target]+IDcol]
# print predictors
alg1 = LinearRegression(normalize=True)
modelfit(alg1, train, test, predictors, target, IDcol, 'alg1.csv')
coef1 = pd.Series(alg1.coef_, predictors).sort_values()
coef1.plot(kind='bar', title='Model Coefficients')


# Random Forest Model 

from sklearn.ensemble import RandomForestRegressor
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg5 = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
modelfit(alg5, train, test, predictors, target, IDcol, 'alg5.csv')
coef5 = pd.Series(alg5.feature_importances_, predictors).sort_values(ascending=False)
coef5.plot(kind='bar', title='Feature Importances')














