# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 21:42:16 2021

@author: Debjit
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


ipl1=pd.read_csv('C:/Users/Debjit/Desktop/Decodr Data Science Course/Projects/Project 4 - IPL  Predictions/IPL-matches-dataset.csv')
ipl2 = pd.read_csv('C:/Users/Debjit/Desktop/Decodr Data Science Course/Projects/Project 4 - IPL  Predictions/IPL-dataset-deliveries.csv')

ipl1.head()
ipl2.head()

ipl1.shape
ipl2.shape

ipl1.columns
ipl2.columns

ipl1['player_of_match'].value_counts()
ipl1['player_of_match'].value_counts()[0:10]

# Makinf a bar-plot  for the Top 5 with most of the man of the match award

plt.figure(figsize=(10,8))
plt.bar(list(ipl1['player_of_match'].value_counts()[0:5].keys()),list(ipl1['player_of_match'].value_counts()[0:5]),color='r')
plt.show()

# Getting the frequency of the rsult column
ipl1['result'].value_counts()

# Finding out the number of Toss wins w.r.t. each teams
ipl1['toss_winner'].value_counts()

# Extracting the records where a team won batting first
batting_first = ipl1[ipl1['win_by_runs']!=0]

# Top 5 teams who won by batting first
batting_first.head()

# Making a Histogram

plt.figure(figsize=(10,6))
plt.hist(batting_first['win_by_runs'])
plt.title('Distribution of Runs')
plt.xlabel('Runs')
plt.show()

# Extracting the record where the team won batting second
batting_second = ipl1[ipl1['win_by_wickets']!=0]

# Top 5 teams who won by batting second
batting_second.head()

# Finding out the number of wins w.r.t each teams after batting first
batting_first['winner'].value_counts().keys()

# Making a bar-plot for Top 3 teams with most number of wins after batting first

plt.figure(figsize=(8,6))
plt.bar(list(batting_first['winner'].value_counts()[0:3].keys()),list(batting_first['winner'].value_counts()[0:3]), color=['blue','yellow','orange'])
plt.show()   

# Making a pie-chart

plt.figure(figsize=(8,6))
plt.pie(list(batting_first['winner'].value_counts()),labels=list(batting_first['winner'].value_counts().keys()), autopct='%0.1f%%')
plt.show()

# Making another figure for the all the team showing the percentage of wins

plt.figure(figsize=(30,8))
sns.countplot(batting_first['winner'],data=ipl1,palette='spring');

df = ipl1[['team1','team2','toss_decision','toss_winner','winner']]
df.dropna(inplace=True);
df

X = df[['team1', 'team2', 'toss_decision', 'toss_winner']]
y = df[['winner']]

all_teams = {}
cnt = 0


for i in range(len(df)):
    if df.loc[i]['team1'] not in all_teams:
        all_teams[df.loc[i]['team1']] = cnt
        cnt += 1
        
    if df.loc[i]['team2'] not in all_teams:
        all_teams[df.loc[i]['team2']] = cnt
        cnt += 1
        
from sklearn.preprocessing import LabelEncoder
teams = LabelEncoder()
teams.fit(all_teams)

encoded_teams = teams.transform(all_teams)

with open('vocab.pkl', 'wb') as f:
    pkl.dump(encoded_teams, f)
with open('inv_vocab.pkl', 'wb') as f:
    pkl.dump(all_teams, f)
    
    

X = np.array(X)
y = np.array(y)

X[:, 0] = teams.transform(X[:, 0])
X[:, 1] = teams.transform(X[:, 1])
X[:, 3] = teams.transform(X[:, 3])

y[:, 0] = teams.transform(y[:, 0])

fb = {'field' : 0, 'bat' : 1}
for i in range(len(X)):
    X[i][2] = fb[X[i][2]] 


X = np.array(X, dtype='int32')
y = np.array(y, dtype='int32')
y_backup = y.copy()

y = y_backup.copy()

ones, zeros = 0,0
for i in range(len(X)):
    if y[i] == X[i][0] :
        if zeros <= 375:
            y[i] = 0
            zeros += 1
        else:
            y[i] = 1
            ones += 1
            t = X[i][0]
            X[i][0] = X[i][1] 
            X[i][1] = t
        
if y[i] == X[i][1] :
        if ones <= 375:
            y[i] = 1
            ones += 1
        else:
            y[i] = 0
            zeros += 1
            t = X[i][0]
            X[i][0] = X[i][1] 
            X[i][1] = t


print(np.unique(y, return_counts=True))



from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.05)


from sklearn.svm import SVC
model1 = SVC().fit(X_train, y_train)
model1.score(X_test, y_test)

from sklearn.tree import DecisionTreeClassifier
model2 = DecisionTreeClassifier().fit(X_train, y_train)
model2.score(X_test, y_test)

from sklearn.ensemble import RandomForestClassifier
model3 = RandomForestClassifier(n_estimators=250).fit(X, y)
model3.score(X_test, y_test)


test = np.array([2,4, 1, 4]).reshape(1,-1)
model1.predict(test)
model2.predict(test)
model3.predict(test)