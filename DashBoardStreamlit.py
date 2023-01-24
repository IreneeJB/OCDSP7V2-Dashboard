import streamlit as st
import myDashboard
import plotly.express as px
import pandas as pd

# Initialize the client API
myclient = myDashboard.ClientAPI("https://projet7.herokuapp.com/",".cache_api.json")
myclient.save_cache(".cache_api.json")

# Create a form with a single input field for the client ID
st.title("Prêt à dépenser - DashBoard")
id_value = st.text_input("Client's ID :", value = "100042", max_chars = 6)

st.sidebar.header("Menu Principal")

# Submit button
# if st.button("Submit"):
# Get client info and prediction
client = myclient.get_client_info(id_value)
pred = myclient.make_client_prediction(id_value)
prets = myclient.get_client_prets(id_value)
prets = pd.DataFrame(prets)
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
        st.text(f"Age : {client_perso_data.loc[:,'AGE'].iloc[0]} ans")
        st.text(f"Sexe : {client_perso_data.loc[:,'CODE_GENDER'].iloc[0]}")
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
        st.text(f"Nombre de jours travaillés : {0-client_pro_data.loc[:,'DAYS_EMPLOYED'].iloc[0]}")
        st.text(f"Type d'emploi : {client_pro_data.loc[:,'OCCUPATION_TYPE'].iloc[0]}")
        st.text(f"Numéro de mobile professionel : {client_pro_data.loc[:,'FLAG_WORK_PHONE'].iloc[0]}")
    
    # Bloc informations de propriété
    with st.expander("Informations de propriété") :
        st.text(f"Propriétaire d'une voiture : {client_poss_data.loc[:,'FLAG_OWN_CAR'].iloc[0]}")
        if client_poss_data.loc[:,'FLAG_OWN_CAR'].iloc[0] == 'Y' :
            st.text(f"Age de la voiture : {client_poss_data.loc[:,'OWN_CAR_AGE'].iloc[0]} ans")
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
    # Charts
    st.title("Critères d'influence sur la solvabilité du client")
    shapdf = pd.DataFrame({'values' : importance[1],
                               'names' : importance[0]})
    fig = px.bar(shapdf, x = 'names', y = 'values', color = 5*['red'] + 5*['green'])
    st.plotly_chart(fig)



if st.sidebar.button("Historique de prêt") :
    st.subheader("Historique de prêts")
    st.markdown("__Prêts en cours__")
    mask = prets.loc[:,"CREDIT_ACTIVE"] == "Active"
    st.text(f"Nombre de prêts ouverts : {prets[mask].shape[0]} prêts.")
    st.text(f"Montant total des prêts en cours : {prets[mask].loc[:,'AMT_CREDIT_SUM'].sum()}")
    st.text(f"Montant à rembourser des prêts en cours : {prets[mask].loc[:,'AMT_CREDIT_SUM_DEBT'].sum()} || {int(10000*prets[mask].loc[:,'AMT_CREDIT_SUM_DEBT'].sum()/prets[mask].loc[:,'AMT_CREDIT_SUM'].sum())/100} %")
    for i in range(len(prets)) :
        pret =  prets.iloc[i:]
        if pret.loc[:,"CREDIT_ACTIVE"].iloc[0] == "Active" :
            with st.expander(f"Prêt ouvert n° {pret.loc[:,'SK_ID_BUREAU'].iloc[0]}   |   Montant : {pret.loc[:,'AMT_CREDIT_SUM'].iloc[0]}.") :
                st.text(f"Ouvert depuis : {0-pret.loc[:,'DAYS_CREDIT'].iloc[0]} jours")
                st.text(f"Montant du prêt : {pret.loc[:,'AMT_CREDIT_SUM'].iloc[0]}")
                st.text(f"Type de prêt : {pret.loc[:,'CREDIT_TYPE'].iloc[0]}")
                st.text(f"Restant à rembourser : {pret.loc[:,'AMT_CREDIT_SUM_DEBT'].iloc[0]}")

                jaugedf = pd.DataFrame({'remboursé' : pret.loc[:,'AMT_CREDIT_SUM'].iloc[0]-pret.loc[:,'AMT_CREDIT_SUM_DEBT'].iloc[0],
                                       'A rembourser' : pret.loc[:,'AMT_CREDIT_SUM_DEBT'].iloc[0]})
                fig = px.bar(jaugedf)
                if pret.loc[:,"CREDIT_DAY_OVERDUE"].iloc[0] != 0 :
                    st.markdown(f"<font color = 'red'> Nombre de jours de retard : {pret.loc[:,'CREDIT_DAY_OVERDUE'].iloc[0]} </font>")
                    st.markdown(f"<font color = 'red'> Montant supplémentaire dû : {pret.loc[:,'AMT_CREDIT_SUM_OVERDUE'].iloc[0]} </font>")

    st.markdown("__Prêts clôturés__")
    mask = prets.loc[:,"CREDIT_ACTIVE"] != "Active"
    st.text(f"Nombre de prêts ouverts : {prets.loc[mask,:].shape[0]} prêts.")
    st.text(f"Montant total des prêts clôturés : {prets.loc[mask,'AMT_CREDIT_SUM'].sum()}")
    for i in range(len(prets)) :
        pret =  prets.iloc[i:]
        if pret.loc[:,"CREDIT_ACTIVE"].iloc[0] != "Active" :
            with st.expander(f"Prêt cloturé  n° {pret.loc[:,'SK_ID_BUREAU'].iloc[0]}   |   Montant : {pret.loc[:,'AMT_CREDIT_SUM'].iloc[0]}.") :
                st.text(f"Ouvert depuis : {0-pret.loc[:,'DAYS_CREDIT'].iloc[0]} jours")
                st.text(f"Montant du prêt : {pret.loc[:,'AMT_CREDIT_SUM'].iloc[0]}")
                st.text(f"Type de prêt : {pret.loc[:,'CREDIT_TYPE'].iloc[0]}")

                if pret.loc[:,"CREDIT_DAY_OVERDUE"].iloc[0] != 0 :
                    st.markdown(f"<font color = 'red'> Nombre de jours de retard : {pret.loc[:,'CREDIT_DAY_OVERDUE'].iloc[0]} </font>")
                    st.markdown(f"<font color = 'red'> Montant supplémentaire dû : {pret.loc[:,'AMT_CREDIT_SUM_OVERDUE'].iloc[0]} </font>")