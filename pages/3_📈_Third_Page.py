import tensorflow_probability as tfp
import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import timedelta
import altair as alt
import streamlit as st 

from tf_keras.models import Model
from tf_keras.layers import Input, LSTM, Dense, Lambda, Concatenate
from tf_keras import backend as K

st.set_page_config(
    page_title="Third Page",
    page_icon="📈",
)

st.markdown("# Real-time Exchange Rate Forecast 📈")

st.markdown('''
            The forecast of the exchange rate on this dashboard page uses Global Forecasting Methods, namely DeepAR. In forecasting, covariates such as inflation and annual GDP of the country are included. Tensorflow and Keras were used in the creation of this model.           
            
            **Objective**: Display the results of real-time exchange rate forecasts using DeepAR for the 5 countries that are the main focus of this dashboard. 
            ''')

long_short = {"Chinese Yuan Renminbi Offshore (CNH)":"CNH",
              "Japanese Yen (JPY)":"JPY",
              "South Korean Won (KRW)":"KRW",
              "Thailand Baht (THB)":"THB",
              "United States Dollar (USD)":"USD"}


def df_build():
  exrate = pd.read_csv("./Exchange rate.csv", index_col=0)
  exrate = exrate.set_index(pd.to_datetime(exrate.index))

  gdp = pd.read_csv("./GDP.csv", index_col=0)
  gdp = gdp.set_index(pd.to_datetime(gdp.index))

  inflation = pd.read_csv("./Inflation.csv", index_col=0)
  inflation = inflation.set_index(pd.to_datetime(inflation.index))

  df = pd.concat([exrate, gdp, inflation], axis=1)
  df = df.ffill()

  return df

df = df_build()

column_temp = st.selectbox(
        label="Choose currency to forecast:",
        options=long_short.keys())

column = long_short[column_temp]

alpha = st.slider("Select confidence level:", 0.0, 1.0, 0.9)
fc_day = st.slider("Select the length of forecasting days:", 1, 30, 15)


# Creating Inference Dataset
def create_inf_dataset(dataframe, count=1000, batch_size=64):
  total_length = len(dataframe)
  df_values = dataframe.values
  df_values = np.expand_dims(df_values, 0)


  dataset = tf.data.Dataset.from_tensor_slices(df_values)
  dataset = dataset.map(lambda window: (window[:, 0, tf.newaxis],
                                        window[-1,0, tf.newaxis, tf.newaxis],
                                        window[:,1:],
                                        tf.reduce_mean(window[:, 0, tf.newaxis],-2,True),
                                        tf.math.reduce_std(window[:, 0, tf.newaxis],-2,True)))
  dataset = dataset.map(lambda window1, window2, window3, mean, var:
                        ((tf.concat([(window1-mean)/var, window3], axis=-1),
                          (window2-mean)/var,
                          mean,
                          var),
                        (mean))
                        )
  dataset = dataset.repeat(count)
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(2)

  return dataset

# Model Builder Function
@st.cache_resource
def model_build():
  enconder_1_latent_dim = 16
  decoder_1_latent_dim = 16

  encoder_1 = LSTM(enconder_1_latent_dim, return_sequences=True, return_state=True)
  decoder_1 = LSTM(decoder_1_latent_dim, return_sequences=True, return_state=True)

  decoder_dense_loc = Dense(1)
  decoder_dense_scale = Dense(1, activation="softplus")

  sampling = tfp.layers.DistributionLambda(
  make_distribution_fn=lambda t: tfp.distributions.Normal(loc=t[0], scale=t[1]),
  convert_to_tensor_fn=tfp.distributions.Distribution.sample)

  concat = Concatenate(axis=-1)


  output_length = 30

  encoder_inputs_inf = Input(shape=(None, 3))
  mean_inputs_inf = Input(shape=(None, 1))
  std_inputs_inf = Input(shape=(None, 1))
  decoder_inputs_inf = Input(shape=(1, 1))

  encoder_1_outputs_inf, state_h_1_inf, state_c_1_inf = encoder_1(encoder_inputs_inf)
  state_h_c_1_inf = [state_h_1_inf, state_c_1_inf]

  all_outputs_inf = []
  inputs_inf = decoder_inputs_inf
  for _ in range(output_length):

    # Run the decoder on one timestep
    decoder_1_outputs_inf, state_h_1_inf, state_c_1_inf = decoder_1(inputs_inf, initial_state=state_h_c_1_inf)

    state_h_c_1_inf = [state_h_1_inf, state_c_1_inf]


    outputs_loc_inf = decoder_dense_loc(decoder_1_outputs_inf)
    outputs_scale_inf = decoder_dense_scale(decoder_1_outputs_inf)

    z_inf = sampling([outputs_loc_inf, outputs_scale_inf])

    # Store the current prediction (we will concatenate all predictions later)

    outputs_inf = concat([z_inf, mean_inputs_inf, std_inputs_inf])

    all_outputs_inf.append(outputs_inf)
    # Reinject the outputs as inputs for the next loop iteration
    # as well as update the states
    inputs_inf = z_inf

  # Concatenate all predictions
  decoder_outputs_inf = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs_inf)

  # Define and compile model as previously
  model_inf = Model([encoder_inputs_inf, decoder_inputs_inf, mean_inputs_inf, std_inputs_inf], decoder_outputs_inf)

  model_inf.load_weights("./model_inference.weights.h5") 

  return model_inf

if st.button("Run"):

  inf_dataset = create_inf_dataset(df[column][-100:])

  model_inf = model_build()

  with st.spinner('Generating the forecast...'):
    inf_prediction = model_inf.predict(inf_dataset)

  sample_prediction = inf_prediction[:,:,0] * inf_prediction[:,:,2] + inf_prediction[:,:,1]

  inf_prediction_upper = np.quantile(sample_prediction, (1+alpha)/2, axis=0)
  inf_prediction_middle = np.quantile(sample_prediction, 0.5, axis=0)
  inf_prediction_lower = np.quantile(sample_prediction, (1-alpha)/2, axis=0)

  source = pd.DataFrame({"Date":pd.bdate_range(df.index.max()+timedelta(days=1), periods=30),
                        "Upper Quantile":inf_prediction_upper,
                        "Median Quantile":inf_prediction_middle,
                        "Lower Quantile":inf_prediction_lower})
  
  source = source[:fc_day]


  band = alt.Chart(source).mark_errorband().encode(
      alt.Y(
          "Upper Quantile:Q",
          scale=alt.Scale(zero=False),
          title="Exchange Rate to IDR"
      ),
      alt.Y2("Lower Quantile:Q"),
      alt.X("Date"),
      tooltip=["Upper Quantile:Q", "Median Quantile:Q", "Lower Quantile:Q", "Date"]
  )

  line = alt.Chart(source).mark_line().encode(
      alt.Y("Median Quantile:Q"),
      alt.X("Date")
  )

  st.altair_chart(band+line, use_container_width = True)
