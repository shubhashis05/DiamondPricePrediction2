import numpy as np
import os,sys
import pandas as pd
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path= os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting Dependent and Independent features from train and test data")
            X_train,y_train,X_test,y_test=(train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])

            models={
                "Linear Regression": LinearRegression(),
                "Lasso" : Lasso(),
                "Ridge" : Ridge(),
                "Elasticnet": ElasticNet(),
                "Decision Tree": DecisionTreeRegressor()
            }
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print("\n.................................................................")

            ## to get the best model score from dict
            best_model_score= max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            ## we got the best model
            best_model = models[best_model_name]
            print(f"Best model found , model name is {best_model_name} and model score is {best_model_score}")
            logging.info(f"Best model found , model name is {best_model_name} and model score is {best_model_score}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.error(f"Error occurred while saving the best model: {e}")
            raise CustomException(e, sys)
