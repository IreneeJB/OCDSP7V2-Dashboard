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

st.sidebar.header("Menu Principal")
st.sidebar.button("Fiche client")
st.sidebar.button("Demande de prêt")
st.sidebar.button("Historique de prêt")

# Submit button
if st.button("Submit"):
    # Get client info and prediction
    client = myclient.get_client_info(id_value)
    pred = myclient.make_client_prediction(id_value)
    prets = myclient.get_client_prets(id_value)
    importance = myclient.get_features_importance(id_value)

    # split client info and prets in tables
    client = pd.DataFrame(client, index = [1])
    client_perso_data = client.loc[:,['CODE_GENDER', 'DAYS_BIRTH', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                                        'CNT_CHILDREN', 'NAME_HOUSING_TYPE', 'REGION_POPULATION_RELATIVE',
                                        'FLAG_MOBIL', 'FLAG_EMAIL', 'CNT_FAM_MEMBERS', ]]
    client_perso_data['DAYS_BIRTH'] = (0-client_perso_data['DAYS_BIRTH'])//365
    client_perso_data = client_perso_data.rename(columns={'DAYS_BIRTH': 'AGE'})
    client_poss_data = client.loc[:,['FLAG_OWN_CAR', 'OWN_CAR_AGE', 'FLAG_OWN_REALTY']]
    client_pro_data = client.loc[:,['AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'DAYS_EMPLOYED', 'FLAG_WORK_PHONE',
                                    'OCCUPATION_TYPE']]


# # Charts
# st.title("Probabilité de solvabilité du client")
# fig = px.pie({'proba': ['yes', 'no'], 'pred': pred}, values='pred', names='proba', )  #color=['#00ff00', '#ff0000'])
# st.plotly_chart(fig)

# st.subheader("Critères d'influence dans le calcul de la probabilité")
# fig = px.pie({'labels': importance[0], 'importance': importance[1]}, values='importance', names='labels')
# st.plotly_chart(fig)
    
    
if st.sidebar.button("Fiche client") :
    st.text(f"\n\t\tFiche Client n°{id_value}")

    # Bloc informations personnelles
    with st.expander("Informations personnelles") :
        st.text(f"Age : {client_perso_data.loc[:,"DAYS_BIRTH"]}")
        st.text(f"Sexe : {client_perso_data.loc[:,"CODE_GENDER"]}")
        st.text(f"Niveau d'éducation : {client_perso_data.loc[:,"NAME_EDUCATION_TYPE"]}")
        st.text(f"Statut familial : {client_perso_data.loc[:,"NAME_FAMILY_STATUS"]}")
        st.text(f"Nombre d'enfants : {client_perso_data.loc[:,"CNT_CHILDREN"]}")
        st.text(f"Type d'habitation : {client_perso_data.loc[:,"NAME_HOUSING_TYPE"]}")
        st.text(f"Numéro de mobile : {client_perso_data.loc[:,"FLAG_MOBIL"]}")
        st.text(f"Adresse e-mail : {client_perso_data.loc[:,"FLAG_EMAIL"]}")

    # Bloc informations professionnelles
    with st.expander("Informations professionnelles") :
        st.text(f"Revenu Annuel : {client_pro_data.loc[:,"AMT_INCOME_TOTAL"]}")
        st.text(f"Type de revenu : {client_pro_data.loc[:,"NAME_INCOME_TYPE"]}")
        st.text(f"Nombre de jours travaillés : {client_pro_data.loc[:,"DAYS_EMPLOYED"]}")
        st.text(f"Type d'emploi : {client_pro_data.loc[:,"OCCUPATION_TYPE"]}")
        st.text(f"Numéro de modile professionel : {client_pro_data.loc[:,"FLAG_WORK_PHONE"]}")
    
    # Bloc informations de propriété
    with st.expander("Informations de propriété") :
        st.text(f"Propriétaire d'une voiture : {client_poss_data.loc[:,"FLAG_OWN_CAR"]}")
        if client_poss_data.loc[:,"FLAG_OWN_CAR"] == 'Y' :
            st.text(f"Age de la voiture : {client_poss_data.loc[:,"OWN_CAR_AGE"]}")
        st.text(f"Propriétaire d'un apparetement : {client_poss_data.loc[:,"FLAG_OWN_REALTY"]}")
