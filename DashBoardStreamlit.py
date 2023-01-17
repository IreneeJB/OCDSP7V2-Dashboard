import streamlit as st
import myDashboard
import plotly.express as px
import pandas as pd

# Initialize the client API
myclient = myDashboard.ClientAPI("https://projet7.herokuapp.com/",".cache_api.json")
myclient.save_cache(".cache_api.json")

# Create a form with a single input field for the client ID
st.header("Home Credit DashBoard")
id_value = st.text_input("Client's ID :")

# Submit button
if st.button("Submit"):
    # Get client info and prediction
    client = myclient.get_client_info(id_value)
    pred = myclient.make_client_prediction(id_value)
    prets = myclient.get_client_prets(id_value)
    
    # Create a pie chart with plotly
    fig = px.pie({'proba': ['yes', 'no'], 'pred': pred, 'colors':['blue','red']}, values='pred', names='proba')
    st.plotly_chart(fig)
    
    # Display client info and prets in tables
    client = pd.DataFrame(client, index = [1])
    client_perso_data = client.loc[:,['CODE_GENDER', 'DAYS_BIRTH', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                                        'CNT_CHILDREN', 'NAME_HOUSING_TYPE', 'REGION_POPULATION_RELATIVE',
                                        'FLAG_MOBIL', 'FLAG_EMAIL', 'CNT_FAM_MEMBERS', ]]
    client_perso_data['DAYS_BIRTH'] = (0-client_perso_data['DAYS_BIRTH'])//365
    client_perso_data = client_perso_data.rename(columns={'DAYS_BIRTH': 'AGE'})
    client_poss_data = client.loc[:,['FLAG_OWN_CAR', 'OWN_CAR_AGE', 'FLAG_OWN_REALTY']]
    client_pro_data = client.loc[:,['AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'DAYS_EMPLOYED', 'FLAG_WORK_PHONE',
                                    'OCCUPATION_TYPE']]
    st.text("Client's personal data")
    st.dataframe(client_perso_data)
    st.text("Client's ownership data")
    st.dataframe(client_poss_data)
    st.text("Client's professional data")
    st.dataframe(client_pro_data)
    st.text("Client's Loans")
    st.dataframe(pd.DataFrame(prets))
