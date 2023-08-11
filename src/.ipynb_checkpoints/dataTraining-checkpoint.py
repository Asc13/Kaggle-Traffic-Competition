import numpy
import pandas
import seaborn
import sys 
import matplotlib.pyplot as plt
from sklearn.model_selection import *
from sklearn.linear_model import *
from sklearn.tree import *
from sklearn.metrics import *
from sklearn.svm import *

TRAINING_DATA_PATH = 'training_data.csv'
TEST_DATA_PATH = 'test_data.csv'

'''
Decision Tree classifier
'''
def decisionTreeClassifierModel(df_training):
    x = df_training.drop(['AVERAGE_SPEED_DIFF'], axis = 1)
    y = df_training['AVERAGE_SPEED_DIFF'].to_frame()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 2021)

    classifier = DecisionTreeClassifier(random_state = 2021)
    classifier.fit(x_train, y_train)
    
    predictions = classifier.predict(x_test)

    print("Accuracy: ", accuracy_score(y_test, predictions))
    print("Precision: ", precision_score(y_test, predictions, average = 'micro'))
    print("Recall: ", recall_score(y_test, predictions, average = 'micro'))

    return predictions

'''
Decision tree regressor
'''
def decisionTreeRegressorModel(df_training):

    x = df_training.drop(['AVERAGE_SPEED_DIFF'], axis = 1)
    y = df_training['AVERAGE_SPEED_DIFF'].to_frame()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 2021)

    regressor = DecisionTreeRegressor(random_state = 2021)
    regressor.fit(x_train, y_train)

    predictions = regressor.predict(x_test)

    print("Accuracy: ", accuracy_score(y_test, predictions))
    print("Precision: ", precision_score(y_test, predictions, average = 'micro'))
    print("Recall: ", recall_score(y_test, predictions, average = 'micro'))

    return predictions

'''
Regressão linear
'''
def linearRegressionModel(df_training):
    
    x = df_training.drop(['AVERAGE_SPEED_DIFF'], axis = 1)
    y = df_training['AVERAGE_SPEED_DIFF'].to_frame()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 2021)

    linear = LinearRegression()
    linear.fit(x_train, y_train)

    predictions = linear.predict(x_test)
    
    print('MAE: ', mean_absolute_error(y_test, predictions))
    print('MSE: ', mean_squared_error(y_test, predictions))
    print('RMSE: ', numpy.sqrt(mean_squared_error(y_test, predictions)))
    return predictions

'''
Regressão logistica
'''
def logisticRegressionModel(df_training):
    x = df_training.drop(['AVERAGE_SPEED_DIFF'], axis = 1)
    x_train, x_test, y_train, y_test = train_test_split(x, df_training.AVERAGE_SPEED_DIFF, test_size = 0.3, random_state = 2021)

    logistic = LogisticRegression()
    logistic.fit(x_train, y_train)
    
    predictions = logistic.predict(x_test)

    #print(classification_report(y_test, predictions))
    
    print("Accuracy: ", accuracy_score(y_test, predictions))
    print("Precision: ", precision_score(y_test, predictions, average = 'micro'))
    print("Recall: ", recall_score(y_test, predictions, average = 'micro'))
 
    return predictions

'''
Support vector machine 
'''
def supportVectorMachineModel(df_training):
    x = df_training.drop(['AVERAGE_SPEED_DIFF'], axis = 1)
    y = df_training['AVERAGE_SPEED_DIFF'].to_frame()
    x_train, x_test, y_train, y_test = train_test_split(x, numpy.ravel(y), test_size = 0.3, random_state = 2021)

    svc = SVC(random_state = 2021)

    svc.fit(x_train, y_train)
    predictions = svc.predict(x_test)

    print("%0.2f accuracy" %(accuracy_score(y_test, predictions)))
    
    return predictions

'''
Grid Search
'''
def gridSearchModel(df_training):
    x_train = df_training.drop(['AVERAGE_SPEED_DIFF'], axis = 1)
    y_train = df_training['AVERAGE_SPEED_DIFF'].to_frame()
    param_grid = {'C':[1,3,5,7], 'gamma':[0.1,0.01,0.001],'kernel':['rbf']}

    grid = GridSearchCV(SVC(random_state = 2021), param_grid, refit = True, verbose = 3)
    grid.fit(x_train,numpy.ravel(y_train))

    return grid

    '''
    x = df_training.drop(['AVERAGE_SPEED_DIFF'], axis = 1)
    y = df_training['AVERAGE_SPEED_DIFF'].to_frame()
    param_grid = {'C':[1,3,5,7], 'gamma':[0.1,0.01,0.001],'kernel':['rbf']}
    x_train, x_test, y_train, y_test = train_test_split(x, numpy.ravel(y), test_size = 0.3, random_state = 2021)

    grid = GridSearchCV(SVC(random_state = 2021), param_grid, refit = True, verbose = 3)
    grid.fit(x_train,y_train)
    
    print(grid.best_params_)
    print(grid.best_estimator_)

    predictions = grid.predict(x_test)
    print(classification_report(y_test, predictions))

    print("%0.2f accuracy" %(accuracy_score(y_test, predictions)))
    return predictions
    '''

def dataTraining(op, df_training): 
    results = []

    if(op == 1):
        results = decisionTreeClassifierModel(df_training)
    elif(op == 2):
        results = decisionTreeRegressorModel(df_training)
    elif(op == 3):
        results = linearRegressionModel(df_training)
    elif(op == 4):
        results = logisticRegressionModel(df_training)
    elif(op == 5):
        results = supportVectorMachineModel(df_training)
    elif(op == 6):
        results = gridSearchModel(df_training)
    
    return results
