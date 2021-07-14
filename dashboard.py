# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 17:17:44 2021

@author: ilyas
"""

##############################################################################
#                       PRET A CONSOMMER DASHBOARD
##############################################################################

import streamlit as st 
import pandas as pd
import numpy as np
import base64
import joblib

# CONFIGURATION DE LA PAGE
############################################################
st.set_page_config(page_title = 'Prêt à dépenser Dashboard', 
    layout='wide')

# FOND D'ECRAN
############################################################
page_bg_img = '''
<style>
.stApp {
background-image: url("https://raw.githubusercontent.com/ITarhouchi/OC_Pret_a_consommer/master/bg.jpg");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# CHARGEMENT DU DATASET TEST
#############################
url_test = 
test_df = pd.read_csv(url_test, index_col=0)
url_train = 
train_df = pd.read_csv(url_train, index_col=0)
url_model = 
model = joblib.load(url_model)


# IDENTIFIANT CLIENT
#########################
### top row 

sk_id_curr, score, decision = st.beta_columns(3)

with sk_id_curr:
    st.markdown("**Identifiant de la demande**")
    code = st.text_input('Entrer SK_ID_CURR')
    if st.button('Simuler la demande'):
        data_client = test_df[test_df['SK_ID_CURR']==code]
        score_value = model.predict_proba(data_client)
        with score:
            st.markdown("**Score**")
            st.write(score_value)
        with decision:
            st.markdown("**Décison**")
            if score_value > 0.5:
                st.write('Demande de prêt acceptée')
            else:
                st.write('Demande de prêt refusée')
        #number1 = 0.1  
    #st.markdown(f"<h1 style='text-align: center; color: grey;'>{number1}</h1>", unsafe_allow_html=True)

#with second_kpi:
#    st.markdown("**Réponse à la demande**")
#    if number1 > 0.5:
#        number2 = 'Accordé'
#    else:
#        number2 = 'Refusé'
#    st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)


