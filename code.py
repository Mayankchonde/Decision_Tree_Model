
import pandas as pd
da=pd.read_csv("D:\Lecture\Data Mining\Lab Exercise 10\Crop_recommendation.csv")

#checking null values
print(da.isnull().sum())

#decribe the data
print(da.describe())

#Plotting  distribution of temperature and ph.
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
sns.distplot(da['temperature'],color="purple",bins=15,hist_kws={'alpha':0.2})
plt.subplot(1, 2, 2)
sns.distplot(da['ph'],color="green",bins=15,hist_kws={'alpha':0.3})

#plotting pair plot
sns.pairplot(da, hue = 'label')

#BOXPLOT
sns.boxplot(y='label',x='ph',data=da)

#boxplot with respect t rainfall
sns.boxplot(y='label',x='P',data=da[da['rainfall']>150])

#boxplot with respect to humidity 
sns.boxplot(y='label',x='P',data=da[da['humidity']>65])

#Data pre-processing
c=da.label.astype('category')
targets = dict(enumerate(c.cat.categories))
da['target']=c.cat.codes
y=da.target
X=da[['N','P','K','temperature','humidity','ph','rainfall']]

#Heatmap
sns.heatmap(X.corr())   

#Feature scaling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# we must apply the scaling to the test set as well that we are computing for the training set
X_test_scaled = scaler.transform(X_test)

#dcision tree model
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
clf.score(X_test,y_test)

#visualize feature importance
import numpy as np
plt.figure(figsize=(10,4), dpi=80)
c_features = len(X_train.columns)
plt.barh(range(c_features), clf.feature_importances_)
plt.xlabel("Feature importance")
plt.ylabel("Feature name")
plt.yticks(np.arange(c_features), X_train.columns)
plt.show()