import yfinance as yf
import pandas as pd
from noaa_sdk import NOAA
from datetime import datetime, timedelta
# from meteostat import Point, Daily
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline


start = datetime(2015,1,1)
end = datetime.now() - timedelta(days = 1)
tickers = ["ZC=F", "WEAT", "SOYB", "DBA", 
            "CL=F", "RB=F", "XLE",
           "TLT", "TIP", "UUP", "CORN", "ZS=F", "WEAT", "ZW=F"]

data = yf.download(tickers, start=start, end=end)
data = data[["Open", "Close", "Volume"]]
corn = data.reset_index()# yf.download("CORN", start = start, end = end)



if isinstance(corn.columns, pd.MultiIndex):
    corn.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in corn.columns]


class CornModel():
    def __init__(self, data):
        self.data = data
        self.model = None
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None

    def engineer_features(self):
        df = self.data.copy()

        df["Close_CL=F1"] = df["Close_CL=F"].shift(1)
        df["Close_UUP1"] = df["Close_UUP"].shift(1)
        df["Close_ZC=F1"] = df["Close_ZC=F"].shift(1)
        df["Close_TIP1"] = df["Close_TIP"].shift(1)
        df["Close_CORN_LAG"] = df["Close_CORN"].shift(1)
        df["Close_DBA_lag"] = df["Close_DBA"].shift(1)
        df["Close_SOYB_lag"] = df["Close_SOYB"].shift(1)
        df["Close_ZS=F1"] =  df["Close_ZS=F"].shift(1)

        df["V_UUP1"] = df["Volume_UUP"].shift(1)
        df["V_ZS=F1"] =  df["Volume_ZS=F"].shift(1)

      
        df["Close_Corn_Future"] = df["Close_CORN"].shift(-5)
        
        df = df.dropna()
        self.data = df




        
    def prep_data(self, test_size = 0.2):
        y = self.data["Close_Corn_Future"]
        x = self.data[["Close_ZS=F1", "Close_CL=F1", "Close_TIP1", "Close_UUP1", "Close_ZC=F1", "Close_CORN_LAG", "Close_DBA_lag", "Close_SOYB_lag", "V_ZS=F1", "V_UUP1"]]
        x = x.dropna()
        y = y.loc[x.index]
        split_index = int(len(x) * 0.8)
        self.x_train, self.x_test = x.iloc[:split_index], x.iloc[split_index:]
        self.y_train, self.y_test = y.iloc[:split_index], y.iloc[split_index:]

        
    def train(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha = 1))
        ])
       
        self.model.fit(self.x_train,self.y_train)

    def evaluate(self):
        y_pred = self.model.predict(self.x_test)
        return {"R2" : r2_score(self.y_test, y_pred),
               "Mean Squared Error": mean_squared_error(self.y_test, y_pred,  squared = False)}
        
    def cross_validate(self, n_splits = 5):
        y = self.data["Close_Corn_Future"]
        x = self.data[["Close_ZS=F1", "Close_CL=F1", "Close_TIP1", "Close_UUP1", "Close_ZC=F1", "Close_CORN_LAG", "Close_DBA_lag", "Close_SOYB_lag", "V_ZS=F1", "V_UUP1"]]
        x = x.dropna() 
        y = y.loc[x.index]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha = 1))
        ])
        
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        r2_scores = cross_val_score(model, x, y, cv = tscv, scoring = "r2")
        return {"R2 scores": r2_scores,
               "Mean r2": np.mean(r2_scores)}







    def predict_weekly(self):
        current_price = corn["Close_CORN"].iloc[-1]
    
        latest_features = self.x_test.iloc[-1:]
        predicted_price = self.model.predict(latest_features)[0]
    
        return {
            "Price at Market Close": float(current_price),
            "Predicted Price (5d Ahead)": float(predicted_price),
            "Expected Change": float(predicted_price - current_price),
            "Expected % Change": float((predicted_price - current_price) / current_price * 100)
        }

    def forecast_5_days(self):
        current_price = self.data["Close_CORN"].iloc[-1]
        latest_features =  self.x_test.iloc[-1:]
        predicted_price = self.model.predict(latest_features)[0]
        preds = np.linspace(current_price, predicted_price, 6)
        
    
        return preds

    def feature_importance_ridge(self):
        ridge = self.model.named_steps["ridge"]
        scaler = self.model.named_steps["scaler"]
        feature_names = self.x_train.columns
    
        # Coefficients correspond to scaled features
        coefs = ridge.coef_
    
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": coefs,
            "Absolute Importance": np.abs(coefs)
        }).sort_values("Absolute Importance", ascending=False)
    
        return importance_df


    def walk_forward_validation(self):
        df = self.data.copy()
        y = df["Close_Corn_Future"]
        X = df[self.x_train.columns]
    
        predictions = []
        actuals = []
    
        for i in range(len(X) - len(self.x_train)):
            train_X = X.iloc[:len(self.x_train) + i]
            train_y = y.iloc[:len(self.x_train) + i]
    
            test_X = X.iloc[len(self.x_train) + i : len(self.x_train) + i + 1]
            test_y = y.iloc[len(self.x_train) + i]
    
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1))
            ])
    
            model.fit(train_X, train_y)
            pred = model.predict(test_X)[0]
    
            predictions.append(pred)
            actuals.append(test_y)
    
        return predictions, actuals


    