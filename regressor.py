#import libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#load data from csv
df = pd.read_csv('electiondata.csv')

#select relevant data
stats = df[['year','dage','rage','dincumb','rincumb','dpincumb','rpincumb','dsenate','dhouse','rhouse','war']]
results = df[['delectoral','relectoral','dstates','rstates','dpopular','rpopular']]
finalstats = stats.iloc[-1]
finalstats = pd.DataFrame([finalstats], columns=stats.columns)
stats = stats.iloc[:-1]
results = results.iloc[:-1]

#create error array
error = np.zeros(results.shape[1])

#test the model on each row
model = LinearRegression()
for x in range(len(stats)):
    newstats = np.delete(stats.to_numpy(), x, axis=0)
    newresults = np.delete(results.to_numpy(), x, axis=0)
    for y in range(results.shape[1]):
        model.fit(newstats, newresults[:, y])
        error[y] += (model.predict(stats.iloc[x].values.reshape(1, -1)).item() - results.iloc[x, y]) ** 2

#calculate RMSE
error = np.sqrt(error / len(stats))

#print RMSE
print(error)

#predict for new data
for y in range(results.shape[1]):
    model.fit(stats, results.to_numpy()[:, y])
    print(model.predict(finalstats).item(), end=" ")