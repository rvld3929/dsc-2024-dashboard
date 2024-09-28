import pandas as pd
import streamlit as st 
import altair as alt
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Second Page",
    page_icon="🕑",
)

st.markdown("# Historical Data of the Exchange Rate, Inflation, and GDP 🕑")

st.markdown('''
            This dashboard page utilizes the datasets of the daily exchange rates [[Kurs Transaksi BI](https://www.bi.go.id/id/statistik/informasi-kurs/transaksi-bi/default.aspx)] (24 January 2001–present) and annual GDP and inflation [[World Development Indicators | DataBank](https://databank.worldbank.org/source/world-development-indicators)] (2000-2023).
            
             **Objective**: Display historical data of exchange rate to IDR, inflation, and GDP for the top 5 countries based on Indonesia's non-oil import contribution. These values will be used in making the forecasting model. 
            ''')


currency_country = {"CNH":"People's Republic of China",
                    "JPY":"Japan",
                    "KRW":"Republic of Korea",
                    "THB":"Thailand",
                    "USD":"United States"}

country_curerency = dict((v,k) for k,v in currency_country.items())


exchange_rate_full = pd.read_csv("./Exchange rate.csv", index_col=0)

inflation_full = pd.read_csv("./Inflation.csv", index_col=0)

gdp_full = pd.read_csv("./GDP.csv", index_col=0)

country = st.selectbox(
    "Please choose a country:",
    country_curerency.keys()
)

currency = country_curerency[country]



max_date = datetime.now()
min_date = max_date - timedelta(days=360*3)

start_date, finish_date = st.date_input(
    "Select a date range:",
    (min_date,max_date))


exchange_rate = exchange_rate_full[start_date.strftime("%Y-%m-%d"):finish_date.strftime("%Y-%m-%d")]
exchange_rate = exchange_rate.rename_axis('Date').reset_index()
exchange_rate["Date"] = pd.to_datetime(exchange_rate["Date"])
exchange_rate = pd.melt(exchange_rate, id_vars="Date", var_name="Currency", value_name="Value")
exchange_rate["Country"] = exchange_rate["Currency"].map(currency_country)

chart1 = alt.Chart(exchange_rate).mark_line().encode(
    x="Date",
    y=alt.Y("Value",scale=alt.Scale(zero=False), title="Exchange Rate to IDR"),
    color=alt.Color("Currency:N").legend(None),
    tooltip=["Date","Currency","Value"]
).transform_filter(
    alt.FieldOneOfPredicate(field='Currency', oneOf=[currency])
)

st.altair_chart(chart1, use_container_width = True)

col1, col2 = st.columns(2)

with col1:
    inflation = inflation_full[start_date.strftime("%Y-%m-%d"):finish_date.strftime("%Y-%m-%d")]
    inflation = inflation.rename_axis('Date').reset_index()
    inflation["Date"] = pd.to_datetime(inflation["Date"])
    inflation['Date'] = inflation['Date'] - pd.Timedelta(days = 365)
    inflation = pd.melt(inflation, id_vars="Date", value_name="Value")
    inflation["Country"] = inflation["variable"].map(currency_country)
    
    chart2 = alt.Chart(inflation).mark_line().encode(
        x="year(Date)",
        y=alt.Y("Value",scale=alt.Scale(zero=False), title="Annual Country Inflation"),
        color=alt.Color("Country:N").legend(None),
        tooltip=["year(Date)","Country","Value"]
        ).transform_filter(
            alt.FieldOneOfPredicate(field='variable', oneOf=[currency])
        )
    
    st.altair_chart(chart2, use_container_width = True)

with col2:

    gdp = gdp_full[start_date.strftime("%Y-%m-%d"):finish_date.strftime("%Y-%m-%d")]
    gdp = gdp.rename_axis('Date').reset_index()
    gdp["Date"] = pd.to_datetime(gdp["Date"])
    gdp['Date'] = gdp['Date'] - pd.Timedelta(days = 365)
    gdp = pd.melt(gdp, id_vars="Date", value_name="Value")
    gdp["Country"] = gdp["variable"].map(currency_country)

    
    chart3 = alt.Chart(gdp).mark_line().encode(
        x="year(Date)",
        y=alt.Y("Value",scale=alt.Scale(zero=False), title="Annual Country GDP ($)"),
        color=alt.Color("Country:N").legend(None),
        tooltip=["year(Date)","Country","Value:Q"]
        ).transform_filter(
            alt.FieldOneOfPredicate(field='variable', oneOf=[currency])
        )
    
    st.altair_chart(chart3, use_container_width = True)
