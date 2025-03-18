#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import pickle


# In[3]:


class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None
    def load_data(self):
        self.data = pd.read_csv(self.file_path) 
    def create_input_output(self, target_column):
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop(target_column, axis=1)



# In[4]:


class ModelHandler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.createModel()
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5
    def labelEncode(self, columns):
        for col in columns:
            if col in self.input_data.columns:
                encoder = LabelEncoder()
                self.data[col] = encoder.fit_transform(self.data[col])
                self.encoders[col] = encoder  
    def duplicateData(self):
        duplikat = self.input_data.duplicated().sum()
        print(f"Data duplication: {duplikat}")        
    def createModel(self, criteria='gini', maxdepth=4):
        self.model = DecisionTreeClassifier(criterion=criteria, max_depth=maxdepth)
    def makePrediction(self):
        self.y_predict = self.model.predict(self.x_test) 
    def createReport(self):
        print('\nClassification Report\n')
        print(classification_report(self.y_test, self.y_predict, target_names=['1','2','3']))
    def split_data(self, test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.input_data, self.output_data, test_size=test_size, random_state=random_state)
    def train_model(self):
        self.model.fit(self.x_train, self.y_train)
    def evaluate_model(self):
        predictions = self.model.predict(self.x_test)
        return accuracy_score(self.y_test, predictions)
    def tuningParameter(self):
        parameters = {
            'criterion': ['gini', 'entropy'],      
            'max_depth': [2, 4, 6, 8, 10],   
        }
        DTClass = DecisionTreeClassifier()
        DTClass= GridSearchCV(DTClass ,
                            param_grid = parameters,   # hyperparameters
                            scoring='accuracy',        # metric for scoring
                            cv=5)
        DTClass.fit(self.x_train,self.y_train)
        print("Tuned Hyperparameters :", DTClass.best_params_)
        print("Accuracy :",DTClass.best_score_)
        self.createModel(criteria =DTClass.best_params_['criterion'],maxdepth=DTClass.best_params_['max_depth'])
    def save_model_to_file(self, filename):
        with open(filename, 'wb') as file:  
            pickle.dump(self.model, file)  


# In[7]:


file_path = 'dataset.csv'  
data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.create_input_output('Species')
input_df = data_handler.input_df
output_df = data_handler.output_df

model_handler = ModelHandler(input_df, output_df)
#Label Encoding
model_handler.labelEncode(['Species']) 
#duplicate data
model_handler.duplicateData()
model_handler.split_data()

print("Before Tuning Parameter")
model_handler.train_model()
print("Model Accuracy:", model_handler.evaluate_model())
model_handler.makePrediction()
model_handler.createReport()
print("After Tuning Parameter")
model_handler.tuningParameter()
model_handler.train_model()
print("Model Accuracy:", model_handler.evaluate_model())
model_handler.makePrediction()
model_handler.createReport()
model_handler.save_model_to_file('iris_model.pkl')


# In[ ]:




