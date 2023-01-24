import pandas as pd
import numpy as np
import missingno as msno
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
from scipy.cluster.hierarchy import dendrogram

##############################################################################
#  initializePandas() :
#         Aucun paramètres
#
#         Initialise les options pandas
# 
#         Return : None
##############################################################################

def initializePandas() :
    pd.set_option('display.max_columns', 10)  # or 1000
    pd.set_option('display.max_rows', 100)  # or 1000
    pd.set_option('display.max_colwidth', 30)  # or 199
    return None

##############################################################################
#  missingValuesInfos(df) :
#         df : pd.dataframe
#
#         Affiche le nombre de valeurs manquantes, totales, le taux de remplissage et la shape du dataframe
#         Affiche la msno.matrix du dataframe          
#
#         Return : None
##############################################################################

def missingValuesInfos(df) :
    '''
    *zeaze
    '''
    nbRows, nbCols = df.shape
    print(f"Il y a {df.isna().sum().sum()} valeurs manquantes sur {nbRows * nbCols} valeurs totales.")
    print(f"Le taux de remplissage est de : {int(((nbRows*nbCols - df.isna().sum().sum())/(nbRows*nbCols))*10000)/100} %")
    print("Dimension du dataframe :",df.shape)
    msno.matrix(df)
    return None

##############################################################################
#  dataFrameInfos(df) :
#         df : pd.dataframe
#
#         Affiche un dataframe d'information sur le df d'entrée          
#
#         Return : None
##############################################################################

def dataFrameInfos(entryDF, num_output = True, bool_output = True, heatmap = True) :

    nbRows, nbCols = entryDF.shape
    print(f"Il y a {entryDF.isna().sum().sum()} valeurs manquantes sur {nbRows * nbCols} valeurs totales.")
    print(f"Le taux de remplissage est de : {int(((nbRows*nbCols - entryDF.isna().sum().sum())/(nbRows*nbCols))*10000)/100} %")
    print("Dimension du dataframe :",entryDF.shape)
    if heatmap :
        msno.matrix(entryDF)
    df = pd.DataFrame(data = None, index = entryDF.columns, columns = ['count','dtype','primary_key','nb_of_NaN','%_of_NaN','n_unique',
                                                                       'unique_most_freq','num_mean','num_median','num_min','num_max',])
    # Stats pour toutes les colonnes
    df['count'] = entryDF.count()
    df['dtype'] = entryDF.dtypes
    df['primary_key'] = entryDF.nunique() == len(entryDF.index)
    df['nb_of_NaN'] = entryDF.isna().sum()
    df['%_of_NaN'] = (entryDF.isna().sum()* 100 / len(entryDF.index)).round(2)
    df['n_unique'] = entryDF.nunique()
    
    for col in entryDF.columns :
        # élément le plus fréquent (si ce n'est pas une clé primaire)
        if df.loc[col,'primary_key'] :
            df.loc[col,'unique_most_freq'] = np.nan
        else :
            df.loc[col,'unique_most_freq'] = entryDF[col].value_counts().sort_values(ascending = False).index[0]
            
        # Ajout des stats sur les chiffres (si demandé) : num_mean, num_median, num_min, num_max 
        if num_output and (entryDF[col].dtype == 'int' or entryDF[col].dtype == 'float') :
            df.loc[col,'num_mean'] = entryDF[col].mean()
            df.loc[col,'num_median'] = entryDF[col].median()
            df.loc[col,'num_std'] = entryDF[col].std()
            df.loc[col,'num_min'] = entryDF[col].min()
            df.loc[col,'num_max'] = entryDF[col].max()
        
        # Ajout des stats sur les booléens (si demandé) : nb_True et %_True
        if bool_output and entryDF[col].dtype == 'bool' :
            nb_True = (entryDF[col] == True).sum()
            df.loc[col,'nb_True'] = nb_True
            df.loc[col,'%_True'] = int(nb_True / len(entryDF[col]) *100)/100
    return df



##############################################################################
#  plot_dendrogram(model, **kwargs) :
#         model : aglomerative clustering
#
#         affiche le dendrogram du modele
#
#         Return : None
##############################################################################



def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    
    
    
##############################################################################
#  
#         
#           NLTK FUNCTIONS
#             - Extraites du notebook d'exemple du projet
#
##############################################################################


# Tokenizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

def tokenizer_fct(sentence) :
    # print(sentence)
    sentence_clean = sentence.replace('-', ' ').replace('+', ' ').replace('/', ' ').replace('#', ' ').replace('_',' ')
    word_tokens = word_tokenize(sentence_clean)
    return word_tokens

# Stop words
from nltk.corpus import stopwords
stop_w = list(set(stopwords.words('english'))) + ['[', ']', ',', '.', ':', '?', '(', ')']
stop_w = stop_w + ['0', '1', '21', '4', '5', '6', '8', '83', '11', '4', '50', '52','6', '63', '64', '69', '74', '8', '85', '9', '99', '10', '1000','1001', '10024982', '101',
                   '1010', '1015', '1033', '1038', '104','1042', '1045', '1049', '107', '1071', '1075', '107750', '108','1080', '1085', '10890', '1092', '1099', '11', '110',
                   '11000026','111', '112', '1132', '1142301', '1143', '1148', '1149', '1150','1153', '1155', '1170', '1183', '119', '1195', '1199', '12', '120',
                   '1200', '121', '1216', '122', '1234', '12399', '125', '126', '129','1290', '1295', '1299', '13', '130', '132', '1333', '13400','1345', '1349', '1350', '137', 
                   '138', '1389', '139', '1395','1399', '14', '140', '1400', '14001', '1410', '144', '1440','1449', '145', '147', '1470', '149', '1490', '1499', '1500',
                   '15000', '150013', '152', '154', '1584', '1599', '16', '160','16116', '161207', '165', '166', '169', '1695', '1697', '1699','17', '170', '1700', '1705', '174', 
                   '1744', '1749', '175','175057', '178', '179', '1790', '1799', '18', '180', '1800', '182','1850', '189', '190', '1900', '194', '195', '1950', '196', 
                   '1975','1979', '1999', '20', '200', '2000', '2004', '2008', '201000','20145', '203', '20564159', '2060', '20600', '209', '2092', '21','210', '2100', '2115', 
                   '212', '213', '214', '2150', '218', '21800','219', '2199', '22', '220', '2200', '22000', '222', '2222', '2244','225', '2250', '226', '228', '229', '23', '230', 
                   '2301', '2345','235', '237', '239', '24', '240', '2400', '241', '244', '24400','2445', '245', '246345', '249', '2495', '2499', '25', '252','2550', '256', 
                   '259', '261', '265', '266', '269', '27', '270','2700', '273', '274', '275', '2750', '275012', '278', '279','2799', '280', '281', '282', '2840', '2849', '285', 
                   '2879', '29','290', '2900', '295', '2950', '296', '297', '2999', '30', '3000','3003', '301', '303', '305', '309', '31', '310', '3125', '315','317', '319', 
                   '32', '320', '325', '3250', '329', '3292', '33','331', '337', '34', '340', '345', '3460', '348', '349', '35','3500', '351', '35390', '354', '355', '356', '36', 
                   '360', '362','3629', '365', '366', '370', '3700', '374', '3749', '379', '38','380', '385', '3860', '388', '38890', '389', '39', '390', '395','398', '3999', 
                   '40', '400', '4032', '404', '405', '407', '410','411', '416', '418', '419', '4200', '4205', '425', '429', '43','430', '435', '439', '440', '444', '447', '449', 
                   '45', '450','4509', '453', '458', '460', '461', '464', '4704', '472', '475','476', '479', '4795', '48', '480', '489', '49', '490', '4989','4995', '4999', '50', 
                   '500', '5000', '50000', '5004', '505','5050', '519', '5199', '520', '523', '5249', '527', '529', '53','535', '539', '5402', '5436', '545', '549', '5499', '55', 
                   '550','552', '5549', '556', '558', '56', '569', '5692', '570', '575','579', '58', '580', '582', '583', '585', '586', '588', '589', '59','590', '5900', '591', 
                   '592', '593', '594', '595', '5999', '600','6000', '6100', '612', '619', '62', '6249', '625', '630', '638','639', '64', '641', '643', '645', '649', '650', 
                   '654', '656', '66','6645', '665', '6655', '669', '675', '676', '677', '680', '6800','685', '689', '69', '6949', '695', '6950290687051', '696', '700','710', 
                   '715', '720', '721', '725', '730', '735', '745', '749','7495', '75', '750', '751', '760', '765', '7668', '769', '770','771', '7760', '78', '780', '7822', 
                   '783', '785', '789', '790','795', '8000', '802', '8100', '812', '816', '819868', '820', '825','825000000000001', '8274', '829', '83', '84', '840', '845', 
                   '849','85', '850', '8500', '8502', '855', '86', '860', '867', '869','870', '878', '88', '880', '89', '8900', '8904214703639', '895',
                   '896', '899', '8999', '90', '900', '9001', '906', '915', '92','921', '925', '927', '929', '939', '94', '940', '9400', '949','950', '960', '965', '968', '97', 
                   '973', '979', '980', '98189','984', '99', '995', '998']

stop_w = stop_w + ['height', 'lengh', 'width', 'inches', 'inch', 'weight', 'flipkart', 'Flipkart','com', 'www', 'http']

def stop_word_filter_fct(list_words) :
    filtered_w = [w for w in list_words if not w in stop_w]
    filtered_w2 = [w for w in filtered_w if len(w) > 2]
    return filtered_w2

# lower case et alpha
def lower_start_fct(list_words) :
    lw = [w.lower() for w in list_words if (not w.startswith("@")) 
    #                                   and (not w.startswith("#"))
                                       and (not w.startswith("http"))]
    return lw

# Lemmatizer (base d'un mot)
from nltk.stem import WordNetLemmatizer

def lemma_fct(list_words) :
    lemmatizer = WordNetLemmatizer()
    lem_w = [lemmatizer.lemmatize(w) for w in list_words]
    return lem_w

# Fonction de préparation du texte pour le bag of words (Countvectorizer et Tf_idf, Word2Vec)
def transform_bow_fct(desc_text) :
    word_tokens = tokenizer_fct(desc_text)
    lw = lower_start_fct(word_tokens)
    sw = stop_word_filter_fct(lw)
    # lem_w = lemma_fct(lw)    
    transf_desc_text = ' '.join(sw)
    return transf_desc_text

# Fonction de préparation du texte pour le bag of words avec lemmatization
def transform_bow_lem_fct(desc_text) :
    word_tokens = tokenizer_fct(desc_text)
    sw = stop_word_filter_fct(word_tokens)
    lw = lower_start_fct(sw)
    lem_w = lemma_fct(lw)    
    transf_desc_text = ' '.join(lem_w)
    return transf_desc_text

# Fonction de préparation du texte pour le Deep learning (USE et BERT)
def transform_dl_fct(desc_text) :
    word_tokens = tokenizer_fct(desc_text)
#    sw = stop_word_filter_fct(word_tokens)
    lw = lower_start_fct(word_tokens)
    # lem_w = lemma_fct(lw)    
    transf_desc_text = ' '.join(lw)
    return transf_desc_text



def stop_word_filter_fct2(list_words, stop_words) :
    filtered_w = [w for w in list_words if not w in stop_words]
    filtered_w2 = [w for w in filtered_w if len(w) > 2]
    return filtered_w2

def transform_bow_fct2(desc_text, stop_words) :
    word_tokens = tokenizer_fct(desc_text)
    sw = stop_word_filter_fct2(word_tokens, stop_words)
    lw = lower_start_fct(sw)
    # lem_w = lemma_fct(lw)    
    transf_desc_text = ' '.join(lw)
    return transf_desc_text