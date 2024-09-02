#import libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet

#load data from csv
df = pd.read_csv('electiondata.csv')

#select relevant data
stats = df[['year','dage','rage','dincumb','rincumb','dpincumb','rpincumb','dsenate','dhouse','rhouse','war']]
results = df[['delectoral','relectoral']]
finalstats = stats.iloc[-1]
finalstats = pd.DataFrame([finalstats], columns=stats.columns)
stats = stats.iloc[:-1]
results = results.iloc[:-1]

#create error array
error = np.zeros(results.shape[1])

#test the model on each row
model = ElasticNet()
for x in range(len(stats)):
    newstats = np.delete(stats.to_numpy(), x, axis=0)
    newresults = np.delete(results.to_numpy(), x, axis=0)
    model.fit(newstats, newresults)
    prediction = model.predict(stats.iloc[x].values.reshape(1, -1))
    error[0] += (prediction[0, 0] - results.iloc[x, 0]) ** 2
    error[1] += (prediction[0, 1] - results.iloc[x, 1]) ** 2

#calculate RMSE
error = np.sqrt(error / len(stats))

#print RMSE
print(error)

#predict for new data
model.fit(stats, results)
print(model.predict(finalstats))