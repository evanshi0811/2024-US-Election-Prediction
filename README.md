# 2024-US-Election-Prediction
Attempting to predict the result of the 2024 US Presidential Election using machine learning algorithms trained on historical election data. This is primarily just a small project to practice machine learning.

I am using election data from 1828-2020 as elections before 1828 had less data, as well as the fact that political parties became were more confusing before 1828, and it is much easier to organize the candadites by political party (also mainly because the election of 1824 had 4 candidates all from the same party).

Since the sample size is rather low, instead of splitting the data into training and testing, I left only 1 election for testing, and trained the model on the remaining data. I repeated this for every election, collecting the RMSE or percent error on how well the model could predict the outcome of the election that was removed for testing.

The models were split into 3 different groups. The first group was univariate regressors, where the RMSE error was calculated for each of the 6 resulting variables. The second group was multivariate regressors, where the RMSE error was calculated only for the delectoral and relectoral variables. Finally, there was classifiers, where percent error was calculated.

Results:
The majority of the models struggled to accurately predict the testing data. The best univariate regressor model had a RMSE of around 0.25 (which is rather high considering the RMSE can range from 0 to 1). The best multivariate regressor model also had a RMSE of around 0.25 (albeit slightly higher). Note, although most of the models were inaccurate, more of them predicted Trump to win the 2024 election. Finally (and incredibly surprisingly), one of the classifier models (MLPClassifier with identity activation and sgd solver), was somehow able to perfectly predict the testing data. I don't understand if this might be due to my own programming error or if I somehow overfitted the model, but either way, the model somehow was 100% accurate when predicting the results of every other election. This model also predicted Trump to win.

So in conclusion, using just the year, age, incumbency, and congressional seatings was likely not enough data to accurately predict the result of a presidential election, as the vast majority of the models struggled (and given the fact that presidential elections are difficult to predict in general). However, one model was able to perfectly predict the results of past presidential elections, which should lead to more research and insight as to if any errors have been made.
