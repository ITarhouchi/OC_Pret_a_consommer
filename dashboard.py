# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 17:17:44 2021

@author: ilyas
"""

##############################################################################
#                       PRET A CONSOMMER DASHBOARD
##############################################################################

# import streamlit as st 

# # Général
# import pandas as pd
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# pd.set_option('display.max_row', 250)
# pd.set_option('display.max_column', 200)

# # Modélisation - Preprocessing
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.compose import make_column_transformer

# # Modélisation - Model
# from sklearn.neighbors import NearestNeighbors

# from io import BytesIO
# import requests

# FONCTIONS UTILSEES DANS LE DASHBOARD
#############################
@st.cache
def load_df(url):
    """Load dataframe from cloud"""
    df = pd.read_csv(url)
    return df

@st.cache
def load_joblib(url):
    """Load joblib file from cloud"""
    model_file = BytesIO(requests.get(url).content)
    file = joblib.load(model_file)    
    return file

@st.cache
def pred(code_client, pip):
    feats = [f for f in test_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    index_client = test_df[test_df['SK_ID_CURR']==code_client].index
    data_client = test_df[feats].loc[index_client, :]
    prediction = pip.predict_proba(data_client)[0][0]
    return prediction
    
@st.cache
def list_feats_imp(pip):
    """List of important features"""
    feats = [f for f in test_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    feature_names = test_df[feats].columns.to_list()
    importances = pip[-1].feature_importances_
    forest_importances = pd.Series(importances, index=feature_names)
    list_feats_imp_tot = forest_importances.sort_values(ascending=False).index.to_list()
    list_feats_imp = [feat for feat in list_feats_imp_tot if feat in num_cols]
    return list_feats_imp

@st.cache(allow_output_mutation=True)
def preprocess(num_cols, cat_cols):
    """Return preprocessing pipeline"""
    model= make_column_transformer((StandardScaler(), num_cols),
                                   (SimpleImputer(strategy='constant', fill_value=0) , cat_cols))
    return model

@st.cache 
def nn_fit(df):
    """Fitting of NearestNeighbors model"""
    feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    df_drop = df[feats]
    df_drop_prep = preprocess(num_cols, cat_cols).fit_transform(df_drop)
    neigh = NearestNeighbors(n_neighbors=20)
    neigh.fit(df_drop_prep)
    return neigh

@st.cache
def nn_client(code_client):
    """Give a random nearestneighbor for target = 0 and target = 1"""
    index_client = test_df[test_df['SK_ID_CURR']==code_client].index
    df_drop = test_df.drop(columns=['TARGET', 'SK_ID_CURR'])
    df_drop_prep = preprocess(num_cols, cat_cols).fit_transform(df_drop)
    data_client = df_drop_prep[index_client]
    neigh = nn_fit(train_df)
    nn_index = neigh.kneighbors(data_client, 20, return_distance=False)
    NN_df = train_df.loc[nn_index[0].tolist(), :]
    NN_df_0 = NN_df[NN_df['TARGET']==0]
    NN_df_1 = NN_df[NN_df['TARGET']==1]    
    sample_0 = NN_df_0.sample(1)
    sample_1 = NN_df_1.sample(1)    
    return sample_0, sample_1

#@st.cache(suppress_st_warning=True)
def stat_plot(n_feats):
    """Plot of Boxplot"""
    mean_style = {"marker":"o","markerfacecolor":"red", "markeredgecolor":"grey"}
    final_feats = list_feats_imp(pip)[:n_feats]
    sns.set()
    if n_feats==5:
        figsize=(15, 7)
    elif n_feats==10:
        figsize=(40, 15)
    fig, ax = plt.subplots(1, n_feats, figsize=figsize)
    for x in range(n_feats):
        sns.boxplot(ax=ax[x], y=train_df[final_feats].iloc[:, x],
                            showmeans = True,
                            showfliers = False,
                            palette='tab10', 
                            meanprops=mean_style)
        ax[x].set(ylabel=None)
        ax[x].set_title(f"{train_df[final_feats].iloc[:, x].name}")
    st.pyplot(fig)

#@st.cache(suppress_st_warning=True)
def credit_plot(credit, n_feats):
    """Plot of max or min credit"""
    final_feats = list_feats_imp(pip)[:n_feats]
    sns.set()
    if n_feats==5:
        figsize=(15, 7)
    elif n_feats==10:
        figsize=(40, 15)
    fig, ax = plt.subplots(1, n_feats, figsize=figsize)
    for x in range(n_feats):
        sns.barplot(ax=ax[x], y = credit[final_feats].iloc[:, x], ci=None)
        ax[x].set(ylabel=None)
        ax[x].set_title(f"{credit[final_feats].iloc[:, x].name}")
    st.pyplot(fig)

#@st.cache(suppress_st_warning=True)
def client_plot(code_client, n_feats):
    """Plot of data client"""
    final_feats = list_feats_imp(pip)[:n_feats]
    sns.set()
    if n_feats==5:
        figsize=(15, 7)
    elif n_feats==10:
        figsize=(40, 15)
    fig, ax = plt.subplots(1, n_feats, figsize=figsize)
    for x in range(n_feats):
        sns.barplot(ax=ax[x], 
                    y=test_df[test_df['SK_ID_CURR']==code_client][final_feats].iloc[:, x],
                    ci=None)
        ax[x].set(ylabel=None)
        ax[x].set_title(f"{test_df[test_df['SK_ID_CURR']==code_client][final_feats].iloc[:, x].name}")
    st.pyplot(fig)

#@st.cache(suppress_st_warning=True)
def client_compare_plot(code_client, n_feats):
    """Plot of data client and compare with other client"""
    final_feats = list_feats_imp(pip)[:n_feats]
    data_client = test_df[test_df['SK_ID_CURR']==code_client][final_feats]
    sample_0, sample_1 = nn_client(code_client)
    data = pd.concat([data_client, sample_1[final_feats], sample_0[final_feats]])
    data_plot = pd.DataFrame(index=['Client', 'Client - ', 'Client +'], 
                              columns=data.columns,
                              data=data.values)
    sns.set()
    if n_feats==5:
        figsize=(15, 7)
    elif n_feats==10:
        figsize=(40, 15)
    fig, ax = plt.subplots(1, n_feats, figsize=figsize)
    for x in range(n_feats):
        sns.barplot(ax=ax[x], 
                    y=data_plot.iloc[:,x].values,
                    x=data_plot.iloc[:, x].index,
                    ci=None)
        #sns.barplot(ax=ax[x], y=sample_0[final_feats].iloc[:, x], color='green')
        #sns.barplot(ax=ax[x], y=sample_1[final_feats].iloc[:, x], color='red')
        ax[x].set(ylabel=None)
        ax[x].set_title(f"{test_df[test_df['SK_ID_CURR']==code_client][final_feats].iloc[:, x].name}")
    st.pyplot(fig)

# CONFIGURATION DE LA PAGE
############################################################
st.set_page_config(page_title = 'Prêt à dépenser Dashboard', 
    layout='wide')

# CHARGEMENT DES DATASETS, LISTES ET MODELES
####################################################
#url_train = "gs://oc_projet_7_test_df/train_df.csv"
url_train = 'train_df_bis.csv'
train_df = load_df(url_train)

#url_test = "gs://oc_projet_7_test_df/test_df.csv"
url_test = 'test_df_bis.csv'
test_df = load_df(url_test)


#url_model = "https://github.com/ITarhouchi/OC_Pret_a_consommer/blob/master/xgb_bestmodel_custom.sav?raw=true"
#pip = load_joblib(url_model)
#model_file = BytesIO(requests.get(url_model).content)
model_file = 'rf_bestmodel_f1.sav'
pip = joblib.load(model_file)

# url_app_num = 
# num_cols = load_joblib(url_app_num)
# url_app_cat = 
# cat_cols = load_joblib(url_app_cat)

num_cols = joblib.load('app_num.sav')
cat_cols = joblib.load('app_cat.sav')

preprocessing = preprocess(num_cols, cat_cols)

# Crédit maximum et minimum
max_credit = train_df[train_df['AMT_CREDIT']==max(train_df['AMT_CREDIT'])]
min_credit = train_df[train_df['AMT_CREDIT']==min(train_df['AMT_CREDIT'])]

# FOND D'ECRAN
############################################################
#page_bg_img = '''
#<style>
#.stApp {
#background-image: url("https://raw.githubusercontent.com/ITarhouchi/OC_Pret_a_consommer/master/bg.jpg");
#background-size: cover;
#}
#</style>
#'''

#st.markdown(page_bg_img, unsafe_allow_html=True)

# IDENTIFIANT / DECISION
#########################
### top row 

sk_id_curr, score, decision = st.beta_columns(3)
dataviz_all, dataviz_client = st.beta_columns(2)
radio_options_1 = st.sidebar.radio('Nombres de variables', ('5', '10'))
select_box_options = st.sidebar.selectbox('Visualisation sur la base de données des clients', 
                                          ('Statistiques générales', 
                                           'Crédit le plus important',
                                           'Crédit le moins important'))
radio_options_2 = st.sidebar.radio('Type de visualisation', ('Client Seul', 'Comparaison avec client similaire'))
n_feats=int(radio_options_1)


with sk_id_curr:
    st.markdown("**Identifiant de la demande**")
    code_select = st.selectbox('Code SK_ID_CURR', test_df['SK_ID_CURR'].values.tolist())
                       # text_input('Entrer SK_ID_CURR (valeurs entre ...')
   # if st.button('Simuler la demande'):
    code_value = int(code_select)
    score_value = pred(code_value, pip)
    with score:
        st.markdown("**Score**")
        st.write("{:.2f}".format(score_value))
    with decision:
        st.markdown("**Décison**")
        if score_value > 0.6:
            st.write('Demande de prêt acceptée')
        else:
            st.write('Demande de prêt refusée')
    with dataviz_all:
        st.markdown("**Données de l'ensemble des clients**")
        if select_box_options == 'Statistiques générales':
            stat_plot(n_feats)            
        elif select_box_options == 'Crédit le plus important':
            credit_plot(max_credit, n_feats)
        elif select_box_options == 'Crédit le moins important':
            credit_plot(min_credit, n_feats)
    with dataviz_client:
        st.markdown("**Données du client**")
        if radio_options_2 == 'Client Seul':
            client_plot(code_value, n_feats)
        elif radio_options_2 == 'Comparaison avec client similaire':
            client_compare_plot(code_value, n_feats)
# DATA VIZ CLIENT ET COMPARAISON
##################################
### top row 

# dataviz_all, dataviz_client = st.beta_columns(2)

# with dataviz_all:
#     st.markdown("**Données de l'ensemble des clients**")
#     radio_options_1 = st.sidebar.radio('Nombres de variables', ('5', '10'))
#     select_box_options = st.sidebar.selectbox('Choix', ('Statistiques générales', 
#                                                         'Crédit le plus important',
#                                                         'Crédit le moins important'))
#     if radio_options_1 == '5':
#         n_feats = 5
#         if select_box_options == 'Statistiques générales':
#             stat_plot(n_feats)            
#         elif select_box_options == 'Crédit le plus important':
#             credit_plot(max_credit, n_feats)
#         elif select_box_options == 'Crédit le moins important':
#             credit_plot(min_credit, n_feats)
#         with dataviz_client:
#             st.markdown("**Données du client**")
#             radio_options_2 = st.sidebar.radio('Type de visualisation', ('Client Seul', 'Comparaison avec client similaire'))
#             if radio_options_2 == 'Client Seul':
#                 client_plot(code_value, n_feats)
#             if radio_options_2 == 'Comparaison avec client similaire':
#                 client_compare_plot(code_value, n_feats)
#     if radio_options_1 == '10':
#         n_feats = 10
#         if select_box_options == 'Statistiques générales':
#             stat_plot(n_feats)            
#         elif select_box_options == 'Crédit le plus important':
#             credit_plot(max_credit, n_feats)
#         elif select_box_options == 'Crédit le moins important':
#             credit_plot(min_credit, n_feats)
#         with dataviz_client:
#             st.markdown("**Données du client**")
#             radio_options_2 = st.sidebar.radio('Type de visualisation', ('Client Seul', 'Comparaison avec client similaire'))
#             if radio_options_2 == 'Client Seul':
#                 client_plot(code_value, n_feats)
#             if radio_options_2 == 'Comparaison avec client similaire':
#                 client_compare_plot(code_value, n_feats)