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



#C:\Users\SHIVAM\.streamlit\config.toml