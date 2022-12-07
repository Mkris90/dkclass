import streamlit as st 
st.markdown(""" This is a Streamlit App """)

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

label_dict = {
    0: "Brandsøgning",
    1: "Informational",
    2: "Inspiration",
    3: "Navigational",
    4: "Transactional"
}
upload_file = st.file_uploader("Choose a file")
model = pickle.load(open("finalized_model.sav","rb"))
if upload_file is not None:
    df = pd.read_csv(upload_file)
    result = {}
    result['Keyword'] = df['Keyword']
    result['volume'] =df['Volume']
    classes =  [label_dict[model.predict(item)[0][0]] for item in df['Keyword'].values ]
    result['Classes'] = classes
    df = pd.DataFrame(result)
    st.download_button(
        label="Download CSV file",
        data=df.to_csv().encode('utf-8'),
        file_name='labbeled_data.csv',
        mime='text/csv'
    )