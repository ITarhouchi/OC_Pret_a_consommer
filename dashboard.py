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

# IDENTIFIANT CLIENT
#########################
### top row 

sk_id_curr, score, decision = st.beta_columns(3)

with sk_id_curr:
    st.markdown("**Identifiant de la demande**")
    text = st.text_input('Entrer SK_ID_CURR')
    if st.button('Simuler la demande'):
        with score:
            st.markdown("**Score**")
            st.write('CA MARCHE')
        with decision:
            st.markdown("**Décison**")
            st.write('1 000 000 €')
        #number1 = 0.1  
    #st.markdown(f"<h1 style='text-align: center; color: grey;'>{number1}</h1>", unsafe_allow_html=True)

#with second_kpi:
#    st.markdown("**Réponse à la demande**")
#    if number1 > 0.5:
#        number2 = 'Accordé'
#    else:
#        number2 = 'Refusé'
#    st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)


