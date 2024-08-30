import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load the data
df = pd.read_csv('electiondata.csv')

# Select the relevant features
stats = df[['year','dage','rage','dincumb','rincumb','dpincumb','rpincumb','dsenate','dhouse','rhouse','war']]

# Split into training data
stats_train = stats.iloc[:-1]
results = df[['delectoral','relectoral','dstates','rstates','dpopular','rpopular']]
results_train = results.iloc[:-1]

# Initialize and train the model
regr = RandomForestRegressor()
regr.fit(stats_train, results_train)  # Pass the DataFrame directly without raveling

# Predict the last row
y_pred = regr.predict(stats.iloc[-1].values.reshape(1, -1))

# Print the prediction
print(y_pred)
