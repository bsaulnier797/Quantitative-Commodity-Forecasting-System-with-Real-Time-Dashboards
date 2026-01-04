import matplotlib
matplotlib.use("Agg")


import io
import re
import time
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, jsonify, make_response, request
import flask
import trading
from trading import CornModel, start, end, corn

app = Flask(__name__)
html_file = corn.to_html()

cm = CornModel(corn)
cm.engineer_features()
cm.prep_data()
cm.train()
cm.evaluate()
market = cm.predict_weekly()
forecast = cm.forecast_5_days()
importance = cm.feature_importance_ridge()
imp = str(importance)
predictions, actual = cm.walk_forward_validation()

@app.route('/')
def home():
    with open("home_html.html") as f:
        html = f.read()

    return html.replace("{market}", str(market))


@app.route('/corn_data.html')
def browse():
    with open("corn_data.html") as f:
        html = f.read()
        
    return html.replace("{html_file}", html_file)


@app.route('/feature_importance.html')
def importance():
    with open("feature_importance.html") as f:
        html = f.read()
        
    return html.replace("{importance}", imp)

@app.route("/dashboard1.svg")
def plot1():
    cn = corn["Close_CORN"]
    soyb = corn["Close_SOYB"]
    date = corn["Date_"]
    
    
    
    
    fig, ax = plt.subplots()
    
    ax.plot(date,cn, label = "Corn Closing Price")
    ax.plot(date, soyb, label = "Soy Bean Closing Price")
    ax.set_xlabel('Date')
    ax.set_ylabel('Closing Price')
    ax.set_title('Corn Price vs. Soy Bean Closing Price')
    ax.legend()
    
    
    fig.savefig("dashboard1.svg", format="svg")
    plt.close(fig)
    
    with open("dashboard1.svg", "r", encoding="utf-8") as f:
        svg_text = f.read()
    return flask.Response(svg_text, headers = {"Content-Type":"image/svg+xml"})

@app.route("/dashboard2.svg")
def plot2():
    fig, ax = plt.subplots()
    
    ax.plot(forecast)
    ax.set_xlabel('Days')
    ax.set_ylabel('Closing Price')
    ax.set_title('5 Day Corn Price Forecast')
    
    
    
    fig.savefig("dashboard2.svg", format="svg")
    plt.close(fig)
    
    with open("dashboard2.svg", "r", encoding="utf-8") as f:
        svg_text = f.read()
    return flask.Response(svg_text, headers = {"Content-Type":"image/svg+xml"})

@app.route("/dashboard3.svg")
def plot3():
    fig, ax = plt.subplots()
    
    ax.scatter(predictions, actual)
    ax.set_xlabel("Predictions")
    ax.set_ylabel("Actual")
    ax.set_title("Walk‑Forward Forecast vs Actual Corn Prices")
    
    fig.savefig("dashboard3.svg", format="svg")
    plt.close(fig)
    
    
    with open("dashboard3.svg", "r", encoding="utf-8") as f:
        svg_text = f.read()
    return flask.Response(svg_text, headers = {"Content-Type":"image/svg+xml"})


if __name__ == "__main__":
    app.run(debug=True)

