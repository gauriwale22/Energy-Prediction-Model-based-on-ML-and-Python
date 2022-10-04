import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

dataset = pd.concat(map(pd.read_csv,["gt_2011.csv","gt_2012.csv",
                                  "gt_2013.csv","gt_2014.csv",
                                  "gt_2015.csv"]),ignore_index=True)

X = df.drop(['TEY'],axis=1)
Y = df.drop(['AT','AP','AH','AFDP','GTEP','TIT','TAT','CDP','CO','NOX'],axis=1)

def xgboosting(X,Y):
	xgbr = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, random_state=7)
	param_grid = {'n_estimators': [200,1000],
          	'max_depth': [3, 8,],
          	'subsample': [0.7,0.8],
          	'learning_rate': (0.01, 0.05),
          	'max_features' : ['sqrt', 'log2']}
	gscv =GridSearchCV(estimator = GradientBoostingRegressor(),
                   	param_grid = param_grid,
                   	cv=2, 
                   	verbose=2)
	gscv.fit(X_train,Y_train)
	gscv.best_params_
	gbr = gscv.best_estimator_
	return gbr

model = xgboosting(X,Y)

# Saving model to disk
pickle.dump(model, open('energy_prediction_model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('energy_prediction_model.pkl','rb'))
print(model.predict([[4.5878,1018.7,83.675,3.5758,23.979,1086.2,549.83,11.898,0.32663,81.952]]))    