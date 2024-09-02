#import libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

#load data from csv
df = pd.read_csv('electiondata.csv')

#select relevant data
stats = df[['dage','rage','dincumb','rincumb','dpincumb','rpincumb','dsenate','dhouse','rhouse','war']]
results = df[['dwin']]
finalstats = stats.iloc[-1].to_frame().T
stats = stats.iloc[:-1]
results = results.iloc[:-1].values.ravel()

#create error variable
error = 0

#test the model on each row
model = DecisionTreeClassifier(random_state=100)
for x in range(len(stats)):
    newstats = stats.drop(index=x).to_numpy()
    newresults = np.delete(results, x, axis=0)
    model.fit(newstats, newresults)
    if model.predict(stats.iloc[x].values.reshape(1, -1)).item() != results[x]:
        error += 1

#calculate percent error
error /= len(stats)

#print percent error
print(error)

#predict for new data
model.fit(stats, results)
print(model.predict(finalstats))