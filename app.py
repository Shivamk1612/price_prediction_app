# pip install streamlit fbprophet yfinance plotly

import streamlit as st
from datetime import date
import yfinance as yf
#from fbprophet import Prophet
#from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2021-01-01"
END = date.today().strftime("%Y-%m-%d")

st.title("Crude Oil Price Prediction App")

crude_oil = ("CL=F", "BZ=F", "QM=F", "UCO", "HCL=F", "BZT=F", "OIL")
selected_oil = st.selectbox("Select dataset for prediction", crude_oil)

n_days = st.slider("Days of prediction:", 5, 10)
period = n_days

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, END)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Loading data.....")
data = load_data(selected_oil)
data_load_state.text("Loading data.....DONE!")

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# XGBoost Prediction

st.subheader('Predicted data')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

df_train2 = data[['Close']]
df_train3 = data[['Date']]

df_train2['target']=df_train2.Close.shift(-1)
df_train3['target']=df_train3.Date.shift(-1)

df_train2.dropna(inplace=True)
df_train3.dropna(inplace=True)

def train_test_split(data,per):
    data=data.values
    n=int(len(data)*(1-per))
    return data[:n],data[n:]

train,test=train_test_split(df_train2,0.2)
train2,test2=train_test_split(df_train3,0.2)

X=train[:,:-1]
Y=train[:,-1]

model=XGBRegressor(obective="reg:squarederror",n_estimator=1000)
model.fit(X,Y)

def xgb_predict(train,val):
    train=np.array(train)
    X,y=train[:,:-1],train[:,-1]
    model=XGBRegressor(objective="reg:squarederror",n_estimators=1000)
    model.fit(X,y)
    
    val =np.array(val).reshape(1,-1)
    pred=model.predict(val)
    return pred[0]

from sklearn.metrics import mean_squared_error

def validate(data,perc):
    predictions=[]
    train,test = train_test_split(data,perc)
    
    history=[x for x in train]
    for i in range(len(test)):
        test_X,test_Y=test[i,:-1],test[i,-1]
        pred=xgb_predict(history,test_X[0])
        predictions.append(pred)
        
        history.append(test[i])
        
    error=mean_squared_error(test[:,-1],predictions)
    
    return error,test[:,-1],predictions


rmse,y,pred=validate(df_train2,0.2)

org = test[:,-1]

date = test2[:,-1]

prediction = np.array(pred)

df = pd.DataFrame({'Date': date, 'Original': org, 'Predicted': prediction})

st.write(df)

def plot_predicted_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'],y=df['Original'], name='original'))
    fig.add_trace(go.Scatter(x=df['Date'],y=df['Predicted'], name='predicted'))
    fig.layout.update(title_text="Predicted Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_predicted_data()



#C:\Users\SHIVAM\.streamlit\config.toml
