import streamlit as st
import myDashboard
import plotly.express as px
import pandas as pd

# Initialize the client API
myclient = myDashboard.ClientAPI("https://projet7.herokuapp.com/",".cache_api.json")
myclient.save_cache(".cache_api.json")

# Create a form with a single input field for the client ID
st.header("Home Credit DashBoard")
id_value = st.text_input("Client's ID :", value = "100042", max_chars = 6)

st.sidebar.header("Menu Principal")
st.sidebar.button("Historique de prêt")

# Submit button
# if st.button("Submit"):
# Get client info and prediction
client = myclient.get_client_info(id_value)
pred = myclient.make_client_prediction(id_value)
prets = myclient.get_client_prets(id_value)
importance = myclient.get_features_importance(id_value)

# split client info and prets in tables
try :
    client = pd.DataFrame(client, index = [1])
    client_perso_data = client.loc[:,['CODE_GENDER', 'DAYS_BIRTH', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                                        'CNT_CHILDREN', 'NAME_HOUSING_TYPE', 'REGION_POPULATION_RELATIVE',
                                        'FLAG_MOBIL', 'FLAG_EMAIL', 'CNT_FAM_MEMBERS', ]]
    client_perso_data['DAYS_BIRTH'] = (0-client_perso_data['DAYS_BIRTH'])//365
    client_perso_data = client_perso_data.rename(columns={'DAYS_BIRTH': 'AGE'})
    client_poss_data = client.loc[:,['FLAG_OWN_CAR', 'OWN_CAR_AGE', 'FLAG_OWN_REALTY']]
    client_pro_data = client.loc[:,['AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'DAYS_EMPLOYED', 'FLAG_WORK_PHONE',
                                    'OCCUPATION_TYPE']]
except : 
    pass
    
if st.sidebar.button("Fiche client") :
    st.subheader(f"\n\t\tFiche Client n°{id_value}")

    # Bloc informations personnelles
    with st.expander("Informations personnelles") :
        st.text(f"Age : {client_perso_data.loc[:,'AGE'].iloc[0]}")
        st.text(f"Sexe : {client_perso_data.loc[:,'CODE_GENDER']}")
        st.text(f"Niveau d'éducation : {client_perso_data.loc[:,'NAME_EDUCATION_TYPE'].iloc[0]}")
        st.text(f"Statut familial : {client_perso_data.loc[:,'NAME_FAMILY_STATUS'].iloc[0]}")
        st.text(f"Nombre d'enfants : {client_perso_data.loc[:,'CNT_CHILDREN'].iloc[0]}")
        st.text(f"Type d'habitation : {client_perso_data.loc[:,'NAME_HOUSING_TYPE'].iloc[0]}")
        st.text(f"Numéro de mobile : {client_perso_data.loc[:,'FLAG_MOBIL'].iloc[0]}")
        st.text(f"Adresse e-mail : {client_perso_data.loc[:,'FLAG_EMAIL'].iloc[0]}")

    # Bloc informations professionnelles
    with st.expander("Informations professionnelles") :
        st.text(f"Revenu Annuel : {client_pro_data.loc[:,'AMT_INCOME_TOTAL'].iloc[0]}")
        st.text(f"Type de revenu : {client_pro_data.loc[:,'NAME_INCOME_TYPE'].iloc[0]}")
        st.text(f"Nombre de jours travaillés : {client_pro_data.loc[:,'DAYS_EMPLOYED'].iloc[0]}")
        st.text(f"Type d'emploi : {client_pro_data.loc[:,'OCCUPATION_TYPE'].iloc[0]}")
        st.text(f"Numéro de modile professionel : {client_pro_data.loc[:,'FLAG_WORK_PHONE'].iloc[0]}")
    
    # Bloc informations de propriété
    with st.expander("Informations de propriété") :
        st.text(f"Propriétaire d'une voiture : {client_poss_data.loc[:,'FLAG_OWN_CAR'].iloc[0]}")
        if client_poss_data.loc[:,'FLAG_OWN_CAR'].iloc[0] == 'Y' :
            st.text(f"Age de la voiture : {client_poss_data.loc[:,'OWN_CAR_AGE'].iloc[0]}")
        st.text(f"Propriétaire d'un apparetement : {client_poss_data.loc[:,'FLAG_OWN_REALTY'].iloc[0]}")

if st.sidebar.button("Demande de prêt") :
    st.subheader("Demande de prêt")
    st.text(f"Montant du prêt : {client.loc[:,'AMT_CREDIT'].iloc[0]}")
    st.text(f"Type de prêt : {client.loc[:,'NAME_CONTRACT_TYPE'].iloc[0]}")
    st.text(f"Rente annuelle : {client.loc[:,'AMT_ANNUITY'].iloc[0]}")

    # Charts
    st.title("Probabilité de solvabilité du client")
    fig = px.pie({'proba': ['yes', 'no'], 'pred': pred}, values='pred', names='proba', color=['#00ff00', '#ff0000'])
    st.plotly_chart(fig)



# # Charts
# st.title("Probabilité de solvabilité du client")
# fig = px.pie({'proba': ['yes', 'no'], 'pred': pred}, values='pred', names='proba', )  #color=['#00ff00', '#ff0000'])
# st.plotly_chart(fig)

# st.subheader("Critères d'influence dans le calcul de la probabilité")
# fig = px.pie({'labels': importance[0], 'importance': importance[1]}, values='importance', names='labels')
# st.plotly_chart(fig)
