import pandas as pd
import streamlit as st 
import altair as alt
from vega_datasets import data
import numpy as np

st.set_page_config(
    page_title="First Page",
    page_icon="🏭",
)

st.write("# Indonesia's Non-Oil and Gas Imports by Country 🏭")

st.markdown('''
            The data used on this dashboard page comes from the Indonesian Ministry of Trade [[Link](https://satudata.kemendag.go.id/data-informasi/perdagangan-luar-negeri/impor-non-migas-negara)].
            ''')


nonoilgasimport_full = pd.read_csv("./Non-oil gas Import.csv")

countries = alt.topo_feature(data.world_110m.url, 'countries')
 
tab1, tab2 = st.tabs(["Map View", "Donut Chart"])

with tab1:

    st.markdown('''
                **Objective**: Display the amount of Indonesia's non-oil and gas imports in million USD from countries around the world using a map from 2019 to 2023.
                ''')

    column1 = st.slider("Choose a year:", 2019, 2023, 2023)
    

    chart1 = (alt.Chart(countries).mark_geoshape().encode(
        color=alt.Color(f'{column1}:Q', title="Million USD"),
        tooltip=['Country:N', alt.Tooltip(f'{column1}:Q', title=f"{column1} (Million USD)")]
        ).transform_lookup(
            lookup='id',
            from_=alt.LookupData(nonoilgasimport_full, 'Numerik', list(nonoilgasimport_full.columns))
        ).project(
            type='equirectangular'
        ))
    
    
    st.altair_chart(chart1, use_container_width = True)

with tab2:

    st.markdown('''
                **Objective**: Display the countries that have the highest contribution to Indonesia's non-oil imports from 2019 to 2023 using donut charts.
                ''')

    column2 = str(st.slider("Choose one of the year:", 2019, 2023, 2023))

    
    top2 = st.slider("Select the number of top countries:", 0, 20, 5)

    nonoilgasimport_full_edit = nonoilgasimport_full[["Country", column2]].sort_values(by=column2, ascending=False)
    nonoilgasimport_full_edit["Percentage (%)"] = np.round(nonoilgasimport_full_edit[column2] * 100 / sum(nonoilgasimport_full_edit[column2]),2)
    _, others_value, others_percentage = nonoilgasimport_full_edit[top2:].sum()
    nonoilgasimport_edit = nonoilgasimport_full_edit[:top2]._append({"Country":"Others", column2:others_value, "Percentage (%)":others_percentage},
                                                            ignore_index=True)
    
    nonoilgasimport_edit["temp"] = np.where(nonoilgasimport_edit["Country"] == "Others", 0, nonoilgasimport_edit[column2]) 

    chart2 = (alt.Chart(nonoilgasimport_edit).mark_arc(innerRadius=50).encode(
        theta=alt.Theta(column2,
                        sort=alt.EncodingSortField("temp", order='descending')),
        color=alt.Color("Country:N",
                        sort=alt.EncodingSortField("temp", order='descending')),
        tooltip=['Country:N', alt.Tooltip(column2, title=f"{column2} (Million USD)"), 'Percentage (%)']
        )).configure_legend(
        orient='right'
        )

    st.altair_chart(chart2, use_container_width = True)

    st.dataframe(nonoilgasimport_full_edit.set_index("Country").rename(columns={column2:f"{column2} (Million USD)"}), use_container_width = True)

