import pandas as pd
from datetime import datetime, timedelta
import requests
import streamlit as st

st.set_page_config(
    page_title="Dashboard",
    page_icon="🌎",
)

st.write("# Exchange Rate Fluctuation Tracker: Real-Time Data & Projections 🌎")

st.markdown('''
            ### *The exchange rate is forecasted using Global Forecasting Models (GFMs) namely DeepAR.* 

            
            The dashboard provides real-time monitoring and forecasting of the exchange rate fluctuations. It offers businesses and decision-makers insights into currency trends, potential risks, and strategic opportunities, helping optimize financial planning, pricing, and risk management in a dynamic global market to encouraging economic and manufacturing growth.                  
            
            There are 3 dashboard pages that can be accessed on the sidebar on the left, namely:
            - 🏭 First Page: Indonesia's Non-Oil and Gas Imports by Country
            - 🕑 Second Page: Historical Data of the Exchange Rate, Inflation, and GDP
            - 📈 Third Page: Real-time Exchange Rate Forecast
            ''')

exchangerate = pd.read_csv("Exchange rate.csv", index_col=0)

start_date = exchangerate.index.max()
end_date = datetime.now().strftime("%Y-%m-%d")

def update_exchangerate(exchangerate):
    df_full = pd.DataFrame()

    try:
        for currency in exchangerate.columns:
            r = requests.post("https://www.bi.go.id/biwebservice/wskursbi.asmx/getSubKursLokal3",
                    {"mts":currency,
                    "startdate":start_date,
                    "enddate":end_date
                    })
            
            er_plus = pd.read_xml(r.content, xpath=".//Table")

            er_plus = pd.DataFrame({"Date":er_plus["tgl_subkurslokal"].str.slice(stop=10),
                "Currency":er_plus["mts_subkurslokal"],
                "Exchange Rate":er_plus["jual_subkurslokal"]})

            er_plus = er_plus.sort_values(by="Date")

            df_full = pd.concat([df_full, er_plus[1:]], axis=0)
    
        df_full = df_full.pivot_table(index="Date", columns="Currency")

        df_full.columns = exchangerate.columns

        exchangerate = pd.concat([exchangerate, df_full],axis=0)

        exchangerate.to_csv("Exchange rate.csv")
    
    except requests.exceptions.ConnectionError:
        st.warning(f"You are not connected to the internet so the forecast feature is not realtime. You can still use this feature with the last exchange rate data on {start_date}.", icon="⚠️")

    except ValueError:
        pass

if start_date != end_date:
    update_exchangerate(exchangerate)




