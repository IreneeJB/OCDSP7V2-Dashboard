import logging
import pandas as pd
import numpy as np
import joblib
from typing import List, Dict, Any,Optional
import statistics
import requests
import json
import pickle

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', datefmt='%m/%d/%Y %H:%M:%S')
log = logging.getLogger()
log.setLevel("DEBUG")
## model def

# class DataBase:
#     def __init__(self):
#         pass

#     def get_id_client(self):
#         pass

#     @classmethod
#     def DataFrame2Json(cls, df:pd.DataFrame)->Dict[str, Any]:
#         if 1 in df.shape:
#             rdict = {}
#             for col in df.columns:
#                 val = df.loc[df.index[0],col]
#                 if isinstance(val, np.int64):
#                     val = int(val)
#                 if isinstance(val, np.float64):
#                     val = float(val)
#                 rdict[col] = val
#             return rdict
#         else: 
#             return [cls.DataFrame2Json(df.loc[[i],:]) for i in df.index]

# class CSV_DataBase(DataBase):
#     def __init__(self, csv_file:str):
#         super().__init__()
#         self.data = pd.read_csv(csv_file)

#     def get_id_client(self, id_client:int)->pd.DataFrame:
#         """take a id_client and return a dataframe
#         """
#         super().get_id_client()
#         client_data = self.data.loc[self.data['SK_ID_CURR'] == id_client,:]
#         if 1 in client_data.shape : 
#             client_data = client_data.to_numpy().reshape(1, -1)
#             client_data = pd.DataFrame(client_data, columns = self.data.columns)
#         return client_data 

#     def get_group(self, id_client:int)->pd.DataFrame:
#         """get a id_client to return a dataframe with list of client with same profil
#         """
#         COLUMNS_GROUP = ["CODE_GENDER",
#                         #"NAME_EDUCATION_TYPE",
#                         #"NAME_FAMILY_STATUS",
#                         #'DAYS_EMPLOYED',
#                         #"DAYS_BIRTH",
#                         "ORGANIZATION_TYPE",
#                         "OCCUPATION_TYPE",
#                        # "NAME_INCOME_TYPE",
#         ]
#         df_c = self.get_id_client(id_client)
#         df = self.data
#         for col in COLUMNS_GROUP:
#             df = df[df[col] == df_c.loc[0,col]]
#         #df = self.data[(self.data[col]==df_c[col] for col in COLUMNS_GROUP)]
#         return df

#     @classmethod
#     def statOnGroup(cls, df:pd.DataFrame): #->Dict[str:Dict[str, float]]:
#         """get a dataframe and make statistics with
#         """
#         COLUMNS_STAT = ["AMT_INCOME_TOTAL",
#                         "DAYS_EMPLOYED",
#         ]
#         rdict = {}
#         for col in COLUMNS_STAT:
#             values = df[col].tolist()
#             #quantiles = statistics.quantiles(values,n=4)  New in version 3.8
#             quantiles = np.quantile(values, [0,0.25,0.5,1])
#             rdict[col] = {"mean":statistics.mean(values),
#                           "q1":quantiles[0],
#                           "q2":quantiles[1],
#                           "q3":quantiles[2],
#                           "q4":quantiles[3],
#                           "min":min(values),
#                           'max':max(values),
#                           }

#         return rdict


# class Model:
#     """load the model 
#     """
#     def __init__(self, path:str, database:DataBase):
#         part1 = joblib.load(path + 'modelPart1.joblib')
#         part2 = joblib.load(path + 'modelPart2.joblib')
#         model_bytes = part1 + part2
#         self.model = pickle.loads(model_bytes)
#         self.database = database
#         del part1, part2, model_bytes

#     def predict(self, input_data:pd.DataFrame)->List[List[float]]:
#         """return the prediction for the input_data
#         """
#         try :
#             prediction = self.model.predict_proba(input_data.drop(['SK_ID_CURR', 'OWN_CAR_AGE', 'OCCUPATION_TYPE'], axis = 1))
#         except ValueError :
#             raise Error
#         log.debug(f'Pred : {prediction}')
#         log.info(f'Pred% : {prediction[0][0]*100 : 0.2f}%')
#         return prediction

#     def predict_id(self, id_client:int)->List[List[float]]:
#         """return the prediction for the client id
#         """
#         log.info(f"Prediction for client id: {id_client}")
#         print(self.database.get_id_client(id_client))
#         return self.predict(self.database.get_id_client(id_client))

#     def get_features_names(self) :
#         transformer = self.model['transformer']
#         onehot_columns = transformer.transformers_[0][1].get_feature_names_out().tolist()
#         target_columns = transformer.transformers_[1][2]
#         scaler_columns = transformer.transformers_[2][2]
#         features_names = onehot_columns + target_columns + scaler_columns
#         return features_names

#     def importance(self, id_client:int) :
#         features_names = self.get_features_names()
#         clients_informations = self.database.get_id_client(id_client)
#         clients_input = self.model['transformer'].transform(clients_informations)
#         clients_input = pd.DataFrame(clients_input, columns = features_names)

#         # feature importance
#         viz = FeatureImportances(self.model['classifier'])
#         viz.fit(clients_input)
#         return viz.features_[-30:], viz.feature_importances_[-30:]
        
class ClientAPI:
    def __init__(self, server:str, cache:str=None):
        """
        :param str cache: path file of json file to load a cache
        """
        if server.endswith("/"):
            server = server[:-1]
        self.server = server
        
        self.cache = {
                "info":{},
                "prediction":{},
                "prets":{},
                "feature_importance":{}
                "group_info" : {}
        }

        if cache:
            self.load_cache(cache, update=True)

    def make_request(self, url:str, method:Optional[str]="get")->Dict[Any, Any]:
        if url.startswith("/"):
            url = url[1:]
        url = f"{self.server}/{url}"
        log.info(f"make reqests to {url}")
        r = requests.get(url)
        if r.status_code != 200:
            log.critical(f"invalid response: {r.status} to {url}")
            return None
        return r.json()

    def get_client_prets(self, id_client:int, using_cache:bool=True)->Dict[str, Any]:
        """using cache (or not), make a requests to get client info
        """
        cache_key = "prets"
        if str(id_client) in self.cache[cache_key] and using_cache:
            log.debug(f"using cache info: {list(self.cache[cache_key].keys())}")
            return self.cache[cache_key][str(id_client)]
        else:
            req = self.make_request(f"/api/v1/client_prets/{id_client}")
            self.cache[cache_key][str(id_client)] = req
            return req

    def get_client_info(self, id_client:int, using_cache:bool=True)->Dict[str, Any]:
        """using cache (or not), make a requests to get client info
        """
        cache_key = "info"
        if str(id_client) in self.cache[cache_key] and using_cache:
            log.debug(f"using cache info: {list(self.cache[cache_key].keys())}")
            return self.cache[cache_key][str(id_client)]
        else:
            req = self.make_request(f"/api/v1/client_info/{id_client}")
            self.cache[cache_key][str(id_client)] = req
            return req

    def make_client_prediction(self, id_client:int, using_cache:bool=True)->Dict[Any, Any]:
        """using cache (or not), make a requests to get client prediction
        """
        cache_key = "prediction"
        if str(id_client) in self.cache[cache_key] and using_cache:
            log.debug(f"using cache prediction: {list(self.cache[cache_key].keys())}")
            return self.cache[cache_key][str(id_client)]
        else:
            req = self.make_request(f"/api/v1/prediction/{id_client}")
            self.cache[cache_key][str(id_client)] = req
            return req

    def get_features_importance(self, id_client:int, using_cache:bool=True)->Dict[Any, Any]:
        """using cache (or not), make a requests to get feature_importance for 1 client
        """
        cache_key = "feature_importance"
        if str(id_client) in self.cache[cache_key] and using_cache:
            log.debug(f"using cache feature_importance: {list(self.cache[cache_key].keys())}")
            return self.cache[cache_key][str(id_client)]
        else:
            req = self.make_request(f"/api/v1/importance/{id_client}")
            self.cache[cache_key][str(id_client)] = req
            return req
    
    def load_cache(self,file, update:bool=False):
        with open(file, 'r') as fp:
            if update:
                self.cache.update(json.load(fp))
            else:
                self.cache = json.load(fp)


    def save_cache(self, file):
        """save the cache to the file
        """
        with open(file, 'w') as fp:
            json.dump(self.cache, fp, indent=4)

    def get_group_infos(self, id_client:int, using_cache:bool=True)->Dict[Any, Any]:
        """using cache (or not), make a requests to get client prediction
        """
        cache_key = "group_info"
        if str(id_client) in self.cache[cache_key] and using_cache:
            log.debug(f"using cache group_info: {list(self.cache[cache_key].keys())}")
            return self.cache[cache_key][str(id_client)]
        else:
            req = self.make_request(f"/api/v1/group_info/{id_client}")
            self.cache[cache_key][str(id_client)] = req
            return req

if __name__ == '__main__':
    mydb = CSV_DataBase('application_test.csv')
    print(DataBase.DataFrame2Json(mydb.get_id_client(125)))
    print(mydb.get_group(125))
    print(mydb.statOnGroup(mydb.get_group(125)))