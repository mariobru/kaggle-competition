#!/usr/bin/env python3 

# Importing the libraries I need
import h2o
import src.dataCleaning as dc
from h2o.automl import H2OAutoML
import pandas as pd

# Loading the dataset into a Pandas dataframe
diamonds =  pd.read_csv('./diamonds-datamad1019/data.csv/data.csv')

# Applying my cleaning pipeline
diamonds = dc.dfCleaning(diamonds)

# Inicialize the h2o environment with the resources I want
h2o.init(nthreads = -1, max_mem_size = 26)

# Load the Pandas DF into h2o
diamonds_h2o=h2o.H2OFrame(diamonds)

# Select the features we want to use for testing our model
x_columns = ["carat","color_num","cut_num","clarity_num","depth"]
y_columns = "price"

# Let's split train and test data
train, test=diamonds_h2o.split_frame(ratios = [.8])
X_train=train[x_columns]
y_train=train[y_columns]
X_test=test[x_columns]
y_test=test[y_columns]

# AutoML
aml_ti = H2OAutoML(max_runtime_secs= 180,max_models= 15, seed= 1, nfolds=0)
aml_ti.train(x = x_columns, y = y_columns, training_frame = train, validation_frame=test)
lb_ti = aml_ti.leaderboard
pred_automl = aml_ti.leader.predict(test)

# Preparing test data for submission
diamonds_test =  pd.read_csv('./diamonds-datamad1019/test.csv')
diamonds_test = dc.dfCleaning(diamonds_test)
diamonds_h2o=h2o.H2OFrame(diamonds_test)
x_columns = ["carat","color_num","cut_num","clarity_num","depth"]
X_train=diamonds_h2o[x_columns]
pred_automl = aml_ti.leader.predict(X_train)
submission = pred_automl.as_data_frame()
submission.rename(columns={"predict": "price"}, inplace=True)
submission.price = submission.price.astype(int)

# Last but not least, we save submission in CSV format
submission.to_csv('./diamonds-datamad1019/submission.csv')