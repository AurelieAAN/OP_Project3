#!/usr/bin/env python
# coding: utf-8

# # Améliorons la santé des français

# -------------------------------------------------------------------------------------------------------------------------------

# # I - Contexte du projet

# ## Problématique : Comment améliorer la santé des français ?
# 

# - Est-ce que le nutriscore y contribue ?
# - Si non, comment pouvons -  nous améliorer la santé de la population ?

# ## Methode :
# - Analyser les caractéristiques du nutriscore
# - Etudier les autres moyens à notre disposition pour améliorer l’efficacité du nutriscore ou apporter de nouvelles solutions

# # II - Présentation de la base de données

# Import des packages que nous allons utiliser tout au long de ce projet

# In[1]:


#notre package de fonctionnalités
from Package import Scripts_Analyse01 as pk

import pandas as pd
import numpy as np
import missingno
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats
from scipy.stats import pearsonr
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from yellowbrick.features import ParallelCoordinates
from plotly.graph_objects import Layout
import jenkspy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale
from sklearn import preprocessing
from scipy.cluster.hierarchy import fcluster


# ## 1) Quels sont nos individus ? Quelles sont nos variables ?

# Import des données provenant du site OpenFood :
# https://world.openfoodfacts.org/

# In[17]:


data = pd.read_csv("./Data/fr.openfoodfacts.org.products.csv", encoding='utf-8', sep="\t",low_memory=False)


# In[18]:


data.shape


# Nous disposons de 320 772 produits alimentaires et de 162 colonnes.

# Regardons le nom des colonnes.

# In[19]:


for i in data.columns:
    print(i)


# Nous disposons du détail des aliments en fonction notamment des apports nutritionnels.
# 

# Etudions un extrait de nos donnéees.

# In[20]:


data.head(5)


# Nous observons beaucoup de données manquantes.

# ## 2) Zoom sur les données manquantes

#  Regardons les variables avec plus de 80 %  de données manquantes.

# In[21]:


tab=pk.del_Nan(data, 0.8,0, 0)


# In[22]:


pd.set_option('display.max_rows', 500)
tab


# Ces variables n'apportent pas suffisament d'information. Nous pouvons les supprimer.

# In[23]:


pk.del_Nan(data, 0.8,1, 2)


# In[24]:


data.shape


# Il nous reste 54 variables.

# Regardons si des variables ne sont pas utiles.

# In[25]:


data.columns


# Certaines variables ne sont pas utiles pour notre analyse comme l'url des fiches ou la date de dernière modification.
# Nous pouvons donc les supprimer et conserver par exemple la date de création de la fiche aliment.
# Voici les colonnes que nous supprimons : 
# url, creator, created_t, image_url, image_small_url, last_modified_t, last_modified_datetime

# In[26]:


data.drop(["url", "creator", "created_t","image_url","image_small_url", "last_modified_t", "last_modified_datetime"], axis=1, inplace=True)


# Ensuite, nous pouvons supprimer les variables avec des données uniques.

# In[27]:


data=pk.data_uniqueone_string(data)


# In[28]:


data.shape


# A présent, nous pouvons tracer la matrice des données manquantes pour mieux observer ces valeurs.

# In[29]:


pk.matrix_vm(data, (16,8), (0.60, 0.64, 0.49))


# Il reste beaucoup de données manquantes. Mais nous pouvons créer des catégories N.A pour les variables qualitatives exceptées pour le nutrition_grade_fr.

# In[30]:


pk.data_fillNA_string(data, "AUTRES", ["nutrition_grade_fr"])


# In[31]:


pk.matrix_vm(data, (14,8), (0.82, 0.28, 0.09))


# Il semble que des colonnes avec plus de 50% de données manquantes soient encore présentes.
# Regardons de plus près ces colonnes.

# In[32]:


missing_value_df = pk.data_missingTab(data)


# In[33]:


def graph_int_bar(data_i, x_i, y_i, x_label_i, y_label_i, palette_color_i, title_i):

    fig = px.bar(ms, x=x_i, y=y_i, color=y_i, 
             labels={x_i:x_label_i,y_i:y_label_i},
              color_continuous_scale=palette_color_i)
    fig.update_layout(
        title_text=title_i, # title of plot
        plot_bgcolor= 'rgba(0, 0, 0, 0)'
    )
    
    fig.show()


# In[34]:


ms =missing_value_df.loc[missing_value_df["percent_missing"]>50]
graph_int_bar(ms, 'column_name', 'percent_missing', "Variables", "% - Valeurs manquantes", "fall", "Variables avec plus de 50% de valeurs manquantes")


# Nous décidons de supprimer ces colonnes.

# In[35]:


pk.del_Nan(data, 0.5,1, 2)


# Regardons les données manquantes restantes.

# In[36]:


missing_value_df = pk.data_missingTab(data)
ms =missing_value_df.loc[missing_value_df["percent_missing"]>0]
graph_int_bar(ms, 'column_name', 'percent_missing', "Variables", "% - Valeurs manquantes", "fall", "Répartition des variables en fonction de leur pourcentage de valeurs manquantes")


# Il reste peu de données manquantes. 

# Supprimons les données qui n'ont aucune colonne quantitatives renseignées (fin _100g). En effet, ces colonnes nous permettront de réaliser notre analyse donc il nous faut au moins une information remplie pour une donnée.

# In[54]:


data=data.loc[(pd.isna(data.energy_100g)==False) | (pd.isna(data.carbohydrates_100g)==False) 
              | (pd.isna(data.proteins_100g)==False) | (pd.isna(data.fat_100g)==False)
        | (pd.isna(data["saturated_fat_100g"])==False) | (pd.isna(data["fiber_100g"])==False)
         | (pd.isna(data["sugars_100g"])==False)
        | (pd.isna(data["salt_100g"])==False)
             | (pd.isna(data["sodium_100g"])==False)]


# In[55]:


data.shape


# Il reste encore des données manquantes, mais les individus apportent suffisamment d'informations pour les conserver. 
# 

# Regardons quelques statistiques descriptives sur nos variables restantes.

# In[56]:


data.describe()


# In[57]:


data[data.select_dtypes(include=['object', 'string']).columns].describe()


# Renommons les variables qui contiennent des tirets.

# In[58]:


data = data.rename(columns={'saturated-fat_100g': 'saturated_fat_100g', 
                       'nutrition-score-fr_100g': 'nutrition_score_fr_100g'})


# In[59]:


data.shape


# Nous avons conservé 41 colonnes.
# Passons au nettoyage de nos données. 

# ## 3) Nettoyage des données

# ### a) Formatage des données

# Vérifions s'ils existent des doublons (toutes les colonnes identiques), mettons d'abord toutes les catégories en majuscule.

# In[41]:


data.head(5)


# In[42]:


data=pk.data_majuscule(data)


# In[43]:


data.head(5)


# ### b) Vérification des doublons

# Regardons s'il y a des lignes dupliquées

# In[44]:


data.duplicated().sum()


# On considère comme produit identique les lignes qui ont le même nom de produit et le même code barre.
# En effet, il est noté que des articles différents peuvent avoir le même code barre.

# In[45]:


data[['code', 'product_name']].duplicated().sum()


# Regardons ces doublons

# In[46]:


data_doublons = data.loc[data[['code', 'product_name']].duplicated(keep=False),:]


# In[47]:


data_doublons


# Ces produits ne sont pas forcément des doublons car ils ne contiennent pas de code identifiant les produits.
# 
# Cependant, Le nom  des produits semblent correspondre au pays de provenance et ces lignes ne contiennent pas beaucoup d'informations. Nous observons beaucoup de valeurs manquantes.
# 
# Nous décidons de supprimer ces lignes afin de ne pas fausser nos résultats car il semble que la saisie de ces produits ait été mal réalisée.

# Faisons une jointure entre notre dataframe de base et notre dataframe qui contient les lignes mal renseignées.

# In[48]:


data_result = pd.merge(data, data_doublons, on=["code", "product_name"], how="outer", indicator=True,  suffixes=('', '_del') )


# Maintenant que nous avons identifié ces lignes dans le dataframe de base, on peut les supprimer.

# In[49]:


data_result = data_result.loc[data_result["_merge"] == "left_only"].drop("_merge", axis=1)


# Conservons les colonnes de notre dataframe de base.

# In[50]:


data_result = data_result[[c for c in data_result.columns if not c.endswith('_del')]]


# In[51]:


data=data_result


# In[52]:


data[['code', 'product_name']].duplicated().sum()


# Passons à la prochaine étape : Etudions chacune de nos variables.

# ## 4) Exploration des données

# ### a) Variables quantitatives

# In[45]:


data.describe().round(2)


# il semble qu'il y a ait des données aberrantes. Nous avons des données négatives pour certaines variables.
# Et nous avons des maximums étonnant comparés au 3ème quartile. Même si l'écart-type n'est pas très important.

# #### Etudes des valeurs aberrantes

# Traçons des stripplots pour toutes nos variables quantitatives. Ces graphiques permettent plus facilement d'observer les valeurs extrêmes pour de grands volumes de données.

# In[46]:


for col in data.select_dtypes(include=['float64']).columns:
    pk.graph_stripplot(data,col, "Nuage de point de la variable "+col,(5,3),"#6D8260")


# Nous observons des données extrêmes dites données aberrantes pour presque toutes nos variables.
# 
# Pour rappel, les valeurs aberrantes sont des données qui ne sont pas anodines pour le jeu de données dont on dispose.
# 
# Commençons par essayer de supprimer ces données grâce aux informations sur internet.
# Nous découvrons que l'aliment le plus calorique contient 900 calories. Nous pouvons donc supprimer les données supérieures ou égales à 901 calories.
# 
# source : https://sante.journaldesfemmes.fr/calories/classement/aliments/calories (provenant de l'Anses)

# In[47]:


data=pk.delete_outliers_UPPER(data, data['energy_100g'], 901)


# Pour l'instant pour chacunes de ces colonnes, nous pouvons supprimer toutes les données supérieures à la portion de 100g et inférieures à 0.

# Nous pourrions calculer l'écart-interquartile, mais celui-ci ne serait pas adapté à ces données. En effet certains aliments peuvent être très proche de 100g et d'autres proche de 0.
# 
# Calculons quand même un exemple pour la variable energy_100g

# In[48]:


pk.outliers(data, data['energy_100g'],0)


# Nous obtenons un écart-interquartile de 1 148 pour la valeur haute (donc largement au-dessus de 901 calories).
# Et nous obtenons une valeur inférieur à 0 pour les valeurs basses. Ce qui n'est pas possible pour les nutritions d'un aliment.
# 
# Ce calcul n'est donc pas adaptées à nos données car elles sont irréelles par rapport aux aliments.

# Supprimons les données supérieures à 100g et inférieurs ou égales à 0 pour les autres "nutriments".

# In[49]:



data=pk.delete_outliers_UPPER(data, data['carbohydrates_100g'], 100)
data=pk.delete_outliers_UPPER(data, data['proteins_100g'], 100)
data=pk.delete_outliers_UPPER(data, data['fat_100g'], 100)
data=pk.delete_outliers_UPPER(data, data['saturated_fat_100g'], 100)
data=pk.delete_outliers_UPPER(data, data['fiber_100g'], 100)
data=pk.delete_outliers_UPPER(data, data['sugars_100g'], 100)
data=pk.delete_outliers_UPPER(data, data['salt_100g'], 100)

data=pk.delete_outliers_LOWER(data, data['energy_100g'], 0)
data=pk.delete_outliers_LOWER(data, data['carbohydrates_100g'], 0)
data=pk.delete_outliers_LOWER(data, data['proteins_100g'], 0)
data=pk.delete_outliers_LOWER(data, data['fat_100g'], 0)
data=pk.delete_outliers_LOWER(data, data['saturated_fat_100g'], 0)
data=pk.delete_outliers_LOWER(data, data['fiber_100g'], 0)
data=pk.delete_outliers_LOWER(data, data['sugars_100g'], 0)
data=pk.delete_outliers_LOWER(data, data['salt_100g'], 0)


# Retraçons nos stripplot.

# In[50]:


for col in data.select_dtypes(include=['float64']).columns:
    pk.graph_stripplot(data,col, "Nuage de point de la variable "+col,(5,3), (0.82, 0.28, 0.09))


# Regardons plus précisément les variables proteins_100g, saturated-fat_100g, fiber_1100g, il semble qu'il reste des valeurs aberrantes.

# #### Commençons par la variable proteins_100g

# In[51]:


def boxplot_int(data_var_i,y_label_i, palette_color_i , mean_i):
    layout = Layout(plot_bgcolor='rgba(0,0,0,0)')

    # Use that layout here
    fig = go.Figure(layout=layout)

    fig.update_xaxes(showline=True, linewidth=2, linecolor='#E3E3E3', gridcolor='#E3E3E3', mirror=True )
    fig.update_yaxes(showline=True, linewidth=2, linecolor='#E3E3E3', gridcolor='#E3E3E3', mirror=True)

    if mean_i==1:
        fig.add_trace(go.Box(y=data_var_i, name=" ",
                        marker_color = palette_color_i, boxmean='sd'))
    else:
        fig.add_trace(go.Box(y=data_var_i, name=" ",
                        marker_color = palette_color_i))
    
    fig.update_layout(
        title_text="Boite à moustache de la variable "+y_label_i, # title of plot
        yaxis_title=y_label_i
    )
    

    
    fig.show()


# In[52]:


boxplot_int(data.proteins_100g, "Proteins_100g", "indianred",0 )


# L'aliment le plus protéiné est : gélatine alimentaire	87,6 g
# 
# 
# source : https://sante.journaldesfemmes.fr/calories/classement/aliments/proteines
# 

# Supprimons les données qui ont des protéines supérieures à 89g.

# In[53]:


data = data.loc[data["proteins_100g"]<=89]


# Regardons à nouveau la boxenplot de la variable proteins_100g.

# In[54]:


boxplot_int(data.proteins_100g, "proteins_100g", "indianred",1 )


# Il reste quelques valeurs extrêmes mais elles sont cohérentes, donc on les conserve pour ne pas perdre d'information.
# 
# Les données ne sont pas très dispersées la boite à moustache est aplatie.

# #### Regardons à présent la variable saturated-fat_100g

# Nous savons que l'aliment qui contient le plus d'acide gras saturés est : pain de friture	92,6 g

# source: https://sante.journaldesfemmes.fr/calories/classement/aliments/acides-gras-satures

# In[55]:


data = data.loc[data["saturated_fat_100g"]<=93]


# In[56]:


def graph_int_violin(data_var_i, fillcolor_i, x_label_i):
    layout = Layout(plot_bgcolor='rgba(0,0,0,0)')

    # Use that layout here
    fig = go.Figure(data=go.Violin(y=data_var_i, box_visible=True, line_color='black',
                               meanline_visible=True, fillcolor=fillcolor_i, opacity=0.6,
                               x0=" "), layout=layout)


    fig.update_xaxes(showline=True, linewidth=2, linecolor='#E3E3E3', gridcolor='#E3E3E3', mirror=True )
    fig.update_yaxes(showline=True, linewidth=2, linecolor='#E3E3E3', gridcolor='#E3E3E3', mirror=True)

    
    fig.update_layout(title_text="Violinplot de la variable "+x_label_i, 
                      yaxis_zeroline=False,
                          yaxis_title=x_label_i)
    fig.show()


# In[57]:


graph_int_violin(data['saturated_fat_100g'], "#6D8260", "saturated_fat_100g")


# Il reste des données extrêmes. Mais elles sont cohérentes. Donc nous allons les conserver.

# #### Etudions la variable fiber_100g

# Pour information, l'aliment le plus riche en fibre est la cannelle avec 43,5g (pour 100g).

# source: https://sante.journaldesfemmes.fr/calories/classement/aliments/fibres

# In[58]:


data = data.loc[data["fiber_100g"]<=45]


# Traçons un violinplot pour observer ces modifications

# In[59]:


graph_int_violin(data['fiber_100g'], "#6D8260", "fiber_100g")


# Nous observons que nos données sont concentrées en bas du violinplot.
# 
# #### Passons à la variable salt_100g

# Traçons un boxenplot qui est comme une boite à moustache sauf qu'il a plus de quantiles.

# In[60]:


pk.graph_boxenplot(data, "salt_100g", (0.82, 0.28, 0.09),"Boxenplot du sel contenu (sur 100g) dans les aliments",(8,6))


# Nous observons que les données sont regroupées à gauche du graphique. Il y a des valeurs très extrêmes.

# L'aliment le plus riche en sel est : sel non iodé non fluoré avec 39100 mg
# 
# source : https://sante.journaldesfemmes.fr/calories/classement/aliments/sel

# Supprimons les données supérieurs à 41g.

# In[61]:


data = data.loc[data["salt_100g"]<=40]


# In[62]:


boxplot_int(data.salt_100g, "salt_100g", "indianred",1 )


# Il reste des valeurs extrêmes mais nous les conservons car elles sont réelles.

# #### Maintenant que nous avons supprimé nos données aberrantes, nous pouvons recalculer nos statistiques et retracer des violinplots

# In[63]:


for col in data.select_dtypes(include=['float64']).columns:
    pk.graph_boxplot(data, col, "Boite à moustache de la variable "+col, "#6D8260", (14,8))


# In[64]:


data.describe().round(2)


# On a des données à -15 pour les nutritions score, mais cela est possible.
# source : https://www.santepubliquefrance.fr/media/files/02-determinants-de-sante/nutrition-et-activite-physique/nutri-score/qr-scientifique-technique-en => voir page 27
# 
# Le min est -15 et le max 40.
# 
# Nous remarquons que les écart-type de nos variables sont assez faibles et que les boites à moustache sont toutes assez resserrées. Ainsi, nos données sont peu dispersées au sein de nos variables quantitatives.

# Nous apprenons plusieurs informations sur nos aliments (sur 100g) :
# - 75% des aliments contiennent moins de 2 additifs et non pas d'huile de palme
# - 50% des aliments apportent plus de 362 calories (energie en calorie)
# - 75 % des aliments contiennent moins de 5g de matières grasses et moins de 1.60g de matières grasses saturées
# - 25% des aliments contiennent moins de 4.20g de carbohydrates
# - 50% des aliments contiennent moins de 3.50g de sucres et moins de 0.46g de sels
# - 25% des aliments ont moins de 0.03g de sodium
# - 75% des aliments ont moins de 1.90g de fibres et moins de 6.59g de protéines
# 
# De plus, nous savons que 75% de nos aliments ont une note au nutriscore fr inférieure à 5 et que le maximum est 32.
# 

# #### Maintenant, étudions les distributions de nos différentes variables.

# In[65]:


def hist_int(data_var_i, color_i, title_i, x_label_i, y_label_i):
    layout = Layout(plot_bgcolor='rgba(0,0,0,0)')
    
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Histogram(
        x=data[data_var_i],
        name='control', # name used in legend and hover labels
        xbins=dict( # bins used for histogram
            start=0,
            end=24,
            size=4
        ),
        marker_color=color_i,
        opacity=0.75
    ))

    fig.update_layout(
        title_text=title_i, # title of plot
        xaxis_title_text=x_label_i, # xaxis label
        yaxis_title_text=y_label_i, # yaxis label
        bargap=0, # gap between bars of adjacent location coordinates
        bargroupgap=0 # gap between bars of the same location coordinates
    )
    

    fig.update_xaxes(showline=True, linewidth=2, linecolor='#E3E3E3', gridcolor='#E3E3E3', mirror=True )
    fig.update_yaxes(showline=True, linewidth=2, linecolor='#E3E3E3', gridcolor='#E3E3E3', mirror=True)
    
    
    fig.show()


# In[66]:


for col in data.select_dtypes(include=['float64']).columns:
    hist_int(col, "#AB2300", "Distribution des aliments en fonction de la variable "+col, col, "Fréquence")


# Nous remarquons que la majorité des distributions des variables sont asymétriques vers la gauche, excepté pour la varible energy_100g qui est plus aplatie.

# #### Réalisons un Test de skewness pour confirmer.
# 
# Pour rappel : 
# - Si y1=0 alors la distribution est symétrique.
# - Si y1>0 alors la distribution est étalée à droite.
# - Si y1<0 alors la distribution est étalée à gauche.

# In[67]:


for col in data.select_dtypes(include=['float64']).columns:
    data_i=data[col].loc[pd.isna(data[col])==False]
    print("Variable : "+col+" ----- y1="+ str(data[col].skew()))


# Le test de Skewness confirme ce que nous savons vu grâce au histogramme, les distributions sont étalées à droite.

# #### Vérifions comment nos variables se comportent par rapport à la loi normale grêce au test de kurtosis.
# 
# Si γ2=0 , alors la distribution a le même aplatissement que la distribution normale.
# 
# Si γ2>0 , alors elle est moins aplatie que la distribution normale : les observations sont plus concentrées.
# 
# Si γ2<0 , alors les observations sont moins concentrées : la distribution est plus aplatie.

# In[68]:


for col in data.select_dtypes(include=['float64']).columns:
    data_i=data[col].loc[pd.isna(data[col])==False]
    print("Variable : "+col+" -- y2="+ str(data[col].kurtosis()))


# Nos données sont toutes plus concentrées que la loi normale.
# 
# #### Faisons un test pour vérifier que nos distributions ne suivent pas une loi normale.

# Pour rappel, les hypothèse de du test de la loi normal est :
# - h0 = la distribution est normale. (p-value > 0.05)
# - h1 = la distribution n'est pas normale. (p-value < 0.05)

# In[69]:


for col in data.select_dtypes(include=['float64']).columns:
    data_i=data[col].loc[pd.isna(data[col])==False]
    print("Variable: "+col+" ----- "+ str(scipy.stats.normaltest(data_i)))


# Nous rejetons H0 pour toutes nos variables, la p-value est inférieure à 0.05.
# Nos distributions ne suivent pas une loi normale.

# ### Créons des regroupements afin que nos distributions soient mieux réparties et que nos variables soient plus pertinentes pour la suite.
# 
# Afin d'améliorer la santé des français, nous savons qu'il y a certains seuils en
# - si l'apport nutritionnel fait partie des aliments dit positifs : -6, -4,-2,0,2,4,6,8,10 ...Etc
# - si l'apport nutritionnel fait partie des aliments dit négatifs : 2,4,6,8,10..Etc

# #### Commençons par la variable additives_n

# In[70]:


def graph_barplot_by_group(data, column, group, color_i):   
    grouped = data.loc[data['nutrition_grade_fr']!="AUTRES"].groupby(['nutrition_grade_fr'], sort=False)
    reg_carbohydrates_counts = grouped[column].value_counts(normalize=True, sort=False)

    occupation_data = [
        {'reg': column, 'nutrition_grade_fr': nutrition_grade_fr, 'percentage': percentage*100} for 
        (nutrition_grade_fr, column), percentage in dict(reg_carbohydrates_counts).items()
    ]

    df_occupation = pd.DataFrame(occupation_data)

    p = sns.barplot(x="reg", y="percentage", hue="nutrition_grade_fr", data=df_occupation)
    _ = plt.setp(p.get_xticklabels(), rotation=90)  # Rotate labels


# In[71]:


pk.graph_hist(data["additives_n"],[0,2,4,6,8,10,24] ,"Distribution des produits en fonction de la variable additives_n",
              "#6D8260", 0,24, 2, 0, 45000, "additives_n", 'Fréquences',(11,7))


# Utilisons l'algorithme de Fisher-Jenks pour détecter les ruptures naturelles. Réalisons 5 groupes comme le nutriscore.

# In[72]:



data_add=data.loc[pd.isna(data['additives_n'])==False]
breaks = jenkspy.jenks_breaks(data_add['additives_n'], nb_class=6)


# In[73]:


breaks=list(set(breaks))
breaks
label = [1,2,3,4,5]


# In[74]:


label


# In[75]:


data.describe()


# In[76]:


data['reg_additives'] = pd.cut(data['additives_n'] , bins=breaks, labels=label, include_lowest=True).to_numpy()


# In[ ]:





# Créons une fonction pour les regroupements, elle utilisera l'algorithme de Fisher-Jenks.

# In[77]:


def reg_fisher_jenks(data, colonne, new_colonne, nb_bin):
    data_add=data.loc[pd.isna(data[colonne])==False]
    breaks = jenkspy.jenks_breaks(data_add[colonne], nb_class=nb_bin)
    label = [1,2,3,4,5]
    #breaks=list(set(breaks))
    data[new_colonne] = pd.cut(data[colonne] , bins=breaks, labels=label[:nb_bin], include_lowest=True).to_numpy()


# In[78]:


pk.graph_barplot(data['reg_additives'], "Répartition des aliments en fonction des additifs", 
              (0.82, 0.28, 0.09),
              0, 70, "Intervalle - additives", "Fréquence en %",70, 1,(11,7))


# In[79]:


pk.graph_barplot(data['reg_additives'], "Répartition des aliments en fonction des additifs", 
              (0.82, 0.28, 0.09),
              0, 70, "Intervalle - additives", "Fréquence en %",70, 1,(11,7))


# In[80]:


graph_barplot_by_group(data, 'reg_additives', 'group', '#6D8260')


# Notre distribution est mieux répartie. Conservons ce regroupement.
# 
# #### Passons à la variable energy_100g

# In[81]:


pk.graph_hist(data["energy_100g"],[0,200,400,600,800,901], "Distribution des produits en fonction de la variable energy_100g","#6D8260",
          0,901, 200, 0, 30000 , "energy_100g", 'Fréquences',(11,7))


# In[82]:


reg_fisher_jenks(data, "energy_100g", "reg_energy", 5)


# In[83]:


pk.graph_barplot(data['reg_energy'], "Répartition des aliments en fonction des calories", 
              (0.82, 0.28, 0.09),
              0, 35, "Classe energie", "Fréquence en %",70, 1,(11,7))


# In[84]:


graph_barplot_by_group(data, 'reg_energy', 'group', '#6D8260')


# Notre distribution est mieux répartie.
# 
# #### Etudions la distribution de la variable fat_100g

# In[85]:


pk.graph_hist(data["fat_100g"],[0,5,10,15,20,25,100], "Distribution des produits en fonction de la variable fat_100g",
              "#6D8260", 0,100, 5, 0, 65000 , "fat_100g", 'Fréquences',(11,7))


# In[86]:


reg_fisher_jenks(data, "fat_100g", "reg_fat", 5)


# In[87]:


pk.graph_barplot(data['reg_fat'], "Répartition des aliments en fonction des matières grasses", 
              (0.82, 0.28, 0.09),
              0, 70, "Classe fat_100g", "Fréquence en %",70, 1,(11,7))


# In[88]:


graph_barplot_by_group(data, 'reg_fat', 'group', '#6D8260')


# Nous validons ce regroupement.
# 
# #### Passons à la variable saturated_fat_100g

# In[89]:


pk.graph_hist(data["saturated_fat_100g"],[0,4,8,90], "Distribution des produits en fonction de la variable fat_100g",
             "#6D8260",
          0,100, 5, 0, 85000 , "saturated_fat_100g", 'Fréquences',(11,7))


# In[90]:


reg_fisher_jenks(data, "saturated_fat_100g", "reg_saturated_fat", 5)


# In[91]:


pk.graph_barplot(data['reg_saturated_fat'], "Répartition des aliments en fonction des matières grasses saturées", 
              (0.82, 0.28, 0.09),
              0, 80, "Note saturated_fat", "Fréquence en %",70, 1,(11,7))


# In[92]:


graph_barplot_by_group(data, 'reg_fat', 'group', '#6D8260')


# Nous conservons ce regroupement.

# #### Etudions la variable carbohydrates_100g

# In[93]:


pk.graph_hist(data["carbohydrates_100g"],[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,100], 
              "Distribution des produits en fonction de la variable carbohydrate_100g",
              "#6D8260",
          0,100, 15, 0, 20000 , "carbohydrates_100g", 'Fréquences',(11,7))


# In[94]:


reg_fisher_jenks(data, "carbohydrates_100g", "reg_carbohydrates", 5)


# In[95]:


pk.graph_barplot(data['reg_carbohydrates'], "Répartition des aliments en fonction des carbohydrates", 
              (0.82, 0.28, 0.09),
              0, 50, "Note carbohydrates", "Fréquence en %",70, 1,(11,7))


# #### Passons à la variable fiber_100g

# In[96]:


pk.graph_hist(data["fiber_100g"],[0,1,2,3,4,5,6,7,8,100], 
              "Distribution des produits en fonction de la variable fiber_100g",
              "#6D8260",
          0,100, 15, 0, 45000 , "fiber_100g", 'Fréquences',(11,7))


# In[97]:


reg_fisher_jenks(data, "fiber_100g", "reg_fiber", 5)


# In[98]:


pk.graph_barplot(data['reg_fiber'], "Répartition des aliments en fonction des fibres", 
              (0.82, 0.28, 0.09),
              0, 65, "Classe fibres", "Fréquence en %",0, 1,(11,7))


# In[280]:


graph_barplot_by_group(data, 'reg_fiber', 'group', '#6D8260')


# #### Regroupons les données au sein de la variable proteins_100g

# In[100]:


pk.graph_hist(data["proteins_100g"],[0,10,15,20,50,100], 
              "Distribution des produits en fonction de la variable proteins_100g","#6D8260",
          0,100, 15, 0, 80000 , "proteins_100g", 'Fréquences',(11,7))


# In[101]:


reg_fisher_jenks(data, "proteins_100g", "reg_protein", 5)


# In[102]:



pk.graph_barplot(data['reg_protein'], "Répartition des aliments en fonction des protéines", 
              (0.82, 0.28, 0.09),
              0, 50, "classe protéine", "Fréquence en %",0, 1,(11,7))


# In[279]:


graph_barplot_by_group(data, 'reg_protein', 'group', '#6D8260')


# Nous utiliserons ce regroupement.

# #### Etudions la variable salt_100g

# In[202]:



pk.graph_hist(data["salt_100g"],[0,2,4,6,8,10,40], "Distribution des produits en fonction de la variable salt_100g","#6D8260",
          0,40, 5, 0, 80000 , "salt_100g", 'Fréquences',(11,7))


# In[203]:


reg_fisher_jenks(data, "salt_100g", "reg_salt", 5)


# In[207]:


pk.graph_barplot(data['reg_salt'], "Répartition des aliments en fonction de leur teneur en sel", 
              (0.82, 0.28, 0.09),
              0, 70, "Classe salt", "Fréquence en %",70, 1,(11,7))


# In[209]:


graph_barplot_by_group(data, 'reg_salt', 'group', '#6D8260')


# ## Conclusion des variables quantitatives

# Nous avons vu que nos données étaient peu dispersées au sein des variables quantitatives.
# Et que les distributions étaient de forme asymétrique vers la gauche.
# 
# Nous avons appris que nos aliments étaient assez bien notés au nutriscore fr. En effet, 75% des aliments ont une note inférieure à 5 sachant que le maximum est 41. 
# 
# Nous avons aussi pu décrire plus précisément les nutriments de nos aliments : 
# - 50% des aliments apportent plus de 362 calories (energie en calorie)
# - 75 % des aliments contiennent moins de 5g de matières grasses et moins de 1.60g de matières grasses saturées
# - 25% des aliments contiennent moins de 4.20g de carbohydrates
# - 50% des aliments contiennent moins de 3.50g de sucres et moins de 0.46g de sels
# - 75% des aliments ont moins de 1.90g de fibres et moins de 6.59g de protéines
# 
# 
# Nous avons dans notre base de données des aliments peu calorique, peu gras et peu salé. Mais ils contiennent peu de fibre et peu de protéines.
# 

# Regardons à présent ce que peut nous apprendre les variables qualitatives sur notre population.

# In[1620]:


data.describe()


# ###  b) Variables qualitatives

# In[1621]:


data.info()


# In[1622]:


data[data.select_dtypes(include=['object', 'string']).columns].describe(include='all').round(2)


# Nous remarquons que la plupart de nos variables qualitatives comportent beaucoup de catégories et que le mode est "N.A" (non renseigné).
# Nous observons aussi que le mode est la modalité B pour le nutriscore Fr et la modalité "NON" pour la variable "ingredients_from_palm_oil".
# 
# Etudions chaque variable pour obtenir plus d'information sur leur répartition.
# 
# 
# 

# #### Commençons par la variable product_name

# Cette variable contient quasiment autant d'individus que de catégories. 
# 
# Regardons le top 10.

# In[1623]:


data["product_name"].value_counts(normalize=True).head(10)


# Comme nous l'avons remarqué précédemment, nos données sont très dispersées au sein de cette variable.

# #### Regardons la variable quantity

# Cette variable contient 3906 catégories. 
# 
# Pour rappel, ces données nous renseigne sur la quantité contenu dans les produits (avec l'unité utilisé)

# In[104]:


data["quantity"].value_counts(normalize=True).head(10)


# La modalité "AUTRES" représente 75% des aliments.
# 
# Retirons cette modalité et regardons à nouveau le top 10.

# In[105]:


data_autres=data.loc[data['quantity']!="AUTRES"]


# In[106]:


data_autres["quantity"].value_counts(normalize=True).head(10)


# 3.7% des aliments contiennent 300g et 1% des aliments contiennent 600g.
# 
# Les données sont très éparpillées au sein des modalités et sont à des échelles différentes. Elle nous apporte pas d'information supplémentaire, nous pouvons la supprimer.
# 
# 

# In[107]:


del data["quantity"]


# #### Etudions les variables packaging et packaging_tags

# La variable packaging contient 4305 catégories et packaging_tag en contient 3592.

# Ces variables contiennent aussi beaucoup de catégorie. Etudions les top 10.

# In[108]:


data["packaging"].value_counts(normalize=True).head(10)


# Les données sont très éparpillées au sein des catégories. Et la variable Autres est surreprésentée pour les 2 catégories (78%)
# Nous pouvons supprimer ces variables

# Excluons la catégorie "AUTRES"

# In[109]:


data_autres=data.loc[data['packaging']!="AUTRES"]
data_autres["packaging"].value_counts(normalize=True).head(10)


# In[110]:


data_autres=data.loc[data['packaging_tags']!="AUTRES"]
data_autres["packaging_tags"].value_counts(normalize=True).head(10)


# Les catégories ne sont pas très pertinentes pour ces deux variables. En effet, nous avons 3% des aliments qui sont dans des conserves et 0.09% des aliments qui sont dans des conserves,métal. Nous pouvons les exclure.

# In[111]:


del data["packaging"]
del data["packaging_tags"]


# In[112]:


data["brands"].nunique()


# In[113]:



max(data_autres["brands"].value_counts(normalize=True))


# Créons une fonction qui supprime les variables qui ont un seuil de catégorie important et qui sont très dispersées.

# In[114]:


def del_quali_norelevant(data,identifiant,  delete_i, del_seuilcat_i, del_seuilfreq_i, all_i):
    info=""
    for col in data.select_dtypes(include=['object']).columns:
        if type(data[col].loc[pd.isna(data[col])==False].iloc[0])== str and col not in identifiant:
                data_autres=data.loc[data[col]!="AUTRES"]
                if delete_i==1 and max(data_autres[col].value_counts(normalize=True).head(10))<del_seuilfreq_i and (data[col].nunique()>del_seuilcat_i):
                    print("deleted "+col)
                    del data[col]
                print("Variable :"+ col + "----Max freq :"+str(max(data_autres[col].value_counts(normalize=True))))      

    


# On supprime les colonnes avec plus de 500 catégorie et dont le maximum de la fréquence d'une modalité est inférieur à 0.05

# In[115]:


del_quali_norelevant(data,["code", "created_datetime", "product_name", "countries", "countries_tags",
                          "ingredients_text"], 1, 500, 0.05, 0)


# In[116]:


data.columns


# In[117]:


data[data.select_dtypes(include=['object', 'string']).columns].describe(include='all').round(2)


# #### Etudions la variable main_category

# In[118]:


data_autres=data.loc[data['main_category']!="AUTRES"]


# In[119]:


data_autres["main_category"].value_counts(normalize=True).head(10)


# In[120]:


cat_top= data["main_category"].value_counts(normalize=True).head(10).reset_index(name="values")


# In[121]:


cat_top["index"]


# In[122]:


data_cat = data_autres.loc[ data['main_category'].isin(cat_top["index"])==True]


# In[123]:


data[data.select_dtypes(include=['object', 'string']).columns].describe(include='all').round(2)


# In[124]:


pk.graph_bubbleplot(data_cat['main_category'], 'g',"main_category", "Fréquence en %", 
                 "Répartition des aliments par catégorie (top 10)", 0, 23,
                    "black", "center", "bold", (14,8),90)


# La partie Autres est surreprésentée (77%).
# 
# Si on exclut la modalité Autres, les données sont très dispersées au sein de cette variable.
# Nous avons 8% des aliments qui appartiennent à la catégorie canned-foods  (aliment en conserve) et 2% des aliments sont des boissons et 1.5% des aliments surgelés.

# #### Regardons la variable ingredients_text

# In[125]:


data["ingredients_text"].head(5)


# Cette variable contient la liste des ingredients contenus dans les aliments.
# Il n'est donc pas pertinent de l'étudier

# #### Etudions les variables states, states_tags et state_fr

# In[126]:


data["states"].unique()


# Ce sont des informations sur l'état de la fiche aliment. Nous pouvons supprimer ces variables.

# In[127]:


data.drop(["states", "states_tags", "states_fr"], axis=1, inplace=True)


# #### Observons les variables additives_tags et	additives_fr

# Cette variable comporte 11 748 catégories

# Regardons le top 10.

# In[128]:


data_autres["additives_tags"].value_counts(normalize=True).head(10)


# In[129]:


data_autres["additives_fr"].value_counts(normalize=True).head(10)


# La variable "additives_fr" est la "traduction" du code des additifs. Elles ont donc une répartition identiques.

# Les données sont très dispersées au sein de ces variables. La catégorie Autres représentent 50% des aliments.
# 
# De plus, nous avons une colonne qui contient le nombre d'additifs, et ces colonnes ne sont pas très parlantes.
# Nous décidons de les supprimer.

# In[130]:


data.drop(["additives_fr", "additives_tags"], axis=1, inplace=True)


# #### Analysons à présent la variable nutrition_grade_fr

# In[131]:


pk.graph_barplot(data['nutrition_grade_fr'], "Répartition des produits selon le nutrition grade", 
              (0.82, 0.28, 0.09),
              0, 40, "nutrition_grade_fr", "Fréquence en %",0,1, (11,7))


# Nous observons que nous avons très peu d'aliments de catégorie E (moins de 5%) et de catégorie D (environs 10%)

# #### Etudions la variable ingredients_from_palm_oil

# Traçons un graphique pour observer la répartition des aliments au sein de cette variable

# In[132]:



t = pd.crosstab(data.ingredients_from_palm_oil, "freq", normalize=True)
t = t.assign(column = t.index, freq = 100 * t.freq)


# In[133]:


fig = px.pie(t, t.column, t.freq , color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()


# Nous remarquons que seulement 0.7% des aliments contiennent potentiellement de l'huile de palme.

# #### Passons aux variables pnns_groups_1 et pnns_groups_2

# Traçons un graphique circulaire pour étudier la répartition des aliments au sein de ces variables.

# In[134]:


import plotly.express as px
import numpy as np
fig = px.sunburst(data, path=['pnns_groups_1', 'pnns_groups_2'])
fig.show()


# Excluons la catégorie "Autres" qui représente 57% des aliments et renommons la catégorie "Unknown" en "AUTRES".

# In[135]:


data['pnns_groups_1'].unique()


# In[136]:


data.head(20)


# In[137]:


data['pnns_groups_1'] = data['pnns_groups_1'].str.replace("UNKNOWN",'AUTRES')
data['pnns_groups_1'] = data['pnns_groups_1'].str.replace("-",' ')
data['pnns_groups_2'] = data['pnns_groups_2'].str.replace("UNKNOWN",'AUTRES')
data['pnns_groups_2'] = data['pnns_groups_2'].str.replace("-",' ')
data_autres=data.loc[(data.pnns_groups_1!="AUTRES") & (data.pnns_groups_2!="AUTRES") ] 


# In[138]:


pk.graph_circle(data["pnns_groups_1"], "pnns_groups_1", "Répartition des aliments en fonction de leur catégorie")


# In[139]:


pk.graph_circle(data_autres["pnns_groups_1"], "pnns_groups_1", "Répartition des aliments en fonction de leur catégorie (hors modalité autres)")


# In[140]:


import plotly.express as px
import numpy as np
fig = px.sunburst(data_autres, path=['pnns_groups_1', 'pnns_groups_2'])
fig.show()


# Nous remarquons que la plupart des aliments sont des boissons et des ingredients.

# #### Décrivons nos données en regardons le nombre d'aliments par date de création des fiches.

# In[141]:



data['year'] = pd.DatetimeIndex(data['created_datetime']).year
data['month'] =pd.DatetimeIndex(data['created_datetime']).month
data["year_month"]=pd.DatetimeIndex(data['created_datetime']).strftime("%Y/%m")


# In[142]:


data


# In[143]:


data.columns


# In[144]:


data_month=data[["year", "month", "nutrition_grade_fr"]].value_counts().reset_index(name="value")


# In[145]:


data_month


# In[146]:




fig = px.scatter(data_month, x="month", y="value", animation_frame="year", animation_group="nutrition_grade_fr",
           color="nutrition_grade_fr", 
           log_x=True, size_max=55)
fig.update_layout(
        title_text="Nombre d'aliments par mois et par an", # title of plot
       # plot_bgcolor= 'rgba(0, 0, 0, 0)'
    )
fig["layout"].pop("updatemenus") # optional, drop animation buttons
fig.show()


# Nous remarquons que la plupart de nos données ont été créées en mars 2017.

# #### Regardons d'où provient la majorité des aliments grâce à la variable countries.

# Nous avons besoin d'importer les pays et leur code iso.

# In[147]:


info_countries=pd.read_json(".\data\countries.json")


# In[148]:


info_countries


# In[149]:


info_countries=pk.data_majuscule(info_countries)


# In[150]:


info_countries = info_countries.rename(columns={'name': 'countries_tags'})


# In[151]:


data.countries_tags.unique()


# In[152]:


data.countries_tags=data.countries_tags.str.replace("EN:","")


# In[153]:


data.countries_tags=data.countries_tags.str.replace("UNITED-STATES","UNITED STATES OF AMERICA")


# In[154]:


data.countries_tags


# !nous pouvons à présent ajouter lke code iso dans notre base de données.

# In[155]:


data_merge = pd.merge(data, info_countries, how="left", on=["countries_tags"], indicator=True,  suffixes=('', '_del'))


# In[156]:


data_merge


# In[157]:


data_merge.alpha3


# In[158]:


data_result = data_merge.loc[data_merge["_merge"] == "both"].drop("_merge", axis=1)
data_result = data_result[[c for c in data_result.columns if not c.startswith('id') |c.startswith('alpha2') | c.endswith('_del') ]]


# In[159]:


data_result.alpha3.unique()


# In[160]:


data= data_result


# In[161]:


data.alpha3.unique()


# In[162]:


maptest=data[["alpha3", "nutrition_grade_fr", "countries_tags"]].value_counts().reset_index(name="value")


# In[163]:


maptest


# In[164]:


import plotly.express as px

fig = px.scatter_geo(maptest, locations="alpha3", color="nutrition_grade_fr",
                     hover_name="countries_tags", size="value",
                     projection="natural earth")
fig.show()


# La majorité de nos aliments sont vendus aux USA.

# #### Analysons à présent la variable nutrition_grade_fr

# In[165]:


pk.graph_barplot(data['nutrition_grade_fr'], "Répartition des produits selon le nutrition grade", 
              (0.82, 0.28, 0.09),
              0, 40, "nutrition_grade_fr", "Fréquence en %",0,1, (11,7))


# Nous observons que nous avons très peu d'aliments de catégorie E (moins de 5%) et de catégorie D (environs 10%)

# In[ ]:





# ## Conclusion qualitatives

# Nous avons appris que les variables qualitatives étaient très disperséesau sein des catégories et que la majorité des aliments étaient dans la catégorie "AUTRES".
# 
# Nous avons aussi remarqué que nos données ont été créées en mars 2017 et que les aliments proviennent des USA.
# Nous savons aussi que nos aliments sont bien notés, 32% des aliments ont un B et 28% des aliments ont un A contre moins de 5% des aliments ont un E.

# Maintenant que nous connaissons nos données. Il est important de pouvoir faire un zoom sur le nutriscore afin d'améliorer la santé des français.

# # III - Est-ce que le nutriscore contribue à améliorer la santé des français ? Dans ce but, est ce que le nutriscore permet de manger équilibrer ?
# 

# ## Nos variables
# 

# Nous allons commencer par sélectionner les colonnes qu'il nous faut pour améliorer la santé des français.
# 
# Certains organismes officiels, comme l’AFSSA (Agence française de sécurité sanitaire des aliments), ont crée des recommandations sous forme d’apports nutritionnels conseillés (ANC) pour chaque type de nutriment.
# 
# source : https://www.vidal.fr/sante/nutrition/equilibre-alimentaire-adulte/recommandations-nutritionnelles-adulte/en-pratique.html
# 
# Selon le tableau nutritionnel, voici les informations dont nous avons besoin : Glucides, Lipides, Protéines, Fibres. Et les aliments qui contiennent de l'huile de palme sont à notifier.
# 
# Comment le Nutri-Score Fr est-il calculé?
# 
# Points négatifs : l'énergie, les graisses saturées, les sucres, et le sel (des niveaux élevés sont considérés comme mauvais pour la santé)
# 
# Points positifs : la proportion de fruits, de légumes, de noix, d'huiles d'olive, de colza et de noix, de fibres et de protéines (les niveaux élevés sont considérés comme bons pour la santé).
# 
# Sélectionnons donc seulement les colonnes nécessaires à notre étude. Nous sélectionnons tous les aliments même ceux qui ne sont pas vendu en France. En effet, ils peuvent être vendus prochainement et être interessant pour notre étude.

# In[166]:


data.columns


# In[167]:


data_study = data[["code", "product_name",  
            "ingredients_text","energy_100g", "carbohydrates_100g", "proteins_100g", "fat_100g",
            "saturated_fat_100g", "fiber_100g", "sugars_100g", "salt_100g",
             "additives_n", "ingredients_from_palm_oil",
            "ingredients_from_palm_oil_n", "ingredients_that_may_be_from_palm_oil_n", 
            "nutrition_grade_fr", "main_category", "nutrition_score_fr_100g", 'reg_additives',
       'reg_energy', 'reg_fat', 'reg_carbohydrates', 'reg_fiber', 
       'reg_protein']]


# In[168]:


data_study=data_study.loc[(pd.isna(data.energy_100g)==False) & (pd.isna(data.carbohydrates_100g)==False) & (pd.isna(data.proteins_100g)==False) & (pd.isna(data.fat_100g)==False)
       & (pd.isna(data["fiber_100g"])==False)& (pd.isna(data["additives_n"])==False)
& (pd.isna(data["nutrition_score_fr_100g"])==False)
                 & (pd.isna(data["nutrition_grade_fr"])==False)
                 & (pd.isna(data["additives_n"])==False)]


# ## Quels sont ses caractéristiques ?
# 
# #### Techniquement, voici les éléments qui agit sur le nutriscore :
# 
# Points négatifs : l'énergie, les graisses saturées, les sucres, et le sel (des niveaux élevés sont considérés comme mauvais pour la santé)
# 
# Points positifs : la proportion de fruits, de légumes, de noix, d'huiles d'olive, de colza et de noix, de fibres et de protéines (les niveaux élevés sont considérés comme bons pour la santé).
# 

# ## 1) Est - ce que si nous mangeons que des aliments avec un nutriscore A, serons - nous en bonne santé ?

# ### Comparons les informations  des différentes notes.

# Excluons les données non renseignées du Nutriscore.

# In[1689]:


dataN = data_study.loc[data["nutrition_grade_fr"]!="AUTRES"].sort_values(by=["nutrition_grade_fr"])


# ### Analyse bivariée : Etudions le lien entre la variable nutrition_grade_fr et les autres variables.

# In[1690]:


dataN.describe()


# In[1691]:


def percentile(n):
    def percentile_(x):
        y=x.loc[pd.isna(x)==False]
        return np.percentile(y, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_


# In[1692]:


for col in data_study.select_dtypes(include=['float64']).columns:
    print(col)
    print(dataN.groupby("nutrition_grade_fr")[col].agg([np.median,np.min, np.max, percentile(25), percentile(75)]))


# In[1693]:


for col in data_study.select_dtypes(include=['float64']).columns:
    print("moyenne "+col +" = "+str(dataN[col].mean()))
    pk.graph_boxplot_by_group(dataN, col,"nutrition_grade_fr", "Boite à moustache de la variable "+col+" en fonction du nutriscore", "rocket_r", (14,8))


# Grâce à ces boites à moustache, nous remarquons qu'ils restent des données extrêmes. Mais dans ce contexte, nous devons les conserver pour ne pas perdre d'information.
# 
# Les boites à moustache sont peu dispersées même si en fonction des notes, leur dispersion peuvent plus ou moins varié.

# Comment pouvons - nous décrire les aliments avec une note A ?
# - riches en fibres et en proteines
# - peu de sel, de sucre et de matières grasses (y compris en graisses saturés)
# - faible en additifs
# - apport en energie proche de la moyenne
# 
# Comment pouvons - nous décrire les aliments avec une note E ?
# - riches en sucre, en additif et en glucides
# - pauvres en fibres, proteines, sel
# - une energie qui varie beaucoup dans les extrêmes.

# In[1694]:


dataN.columns


# In[2906]:


sns.catplot(x = "nutrition_grade_fr", hue = "ingredients_from_palm_oil", data = dataN, kind = "count", palette="rocket")


# Les produits qui contiennent potentiellement de l'huile de palme ont les meilleurs notes.

# Confirmons nos résultats précédents en vérifiant le lien entre nos variables quantitatives et le grade du nutriscore

# ## 2) Statistiques inférentielles : ANOVA et Test de Krusdall-Wallis

# ## a) Essayons de modéliser nos données avec une ANOVA

# Pour cela, nous pourrions modéliser nos données en réalisant une ANOVA.
# Mais nous devons vérifier 3 hypothèses : 
# - l'indépendance entre chaque groupe
# - l'égalité des variances
# - la normalité des résidus (cela permet de ne pas affirmer qu'il existe une différence de moyenne entre les groupes qui serait causée par le hasard).

# ### L'indépendance entre nos groupes

# Selon le contexte, nous savons que chaque lettre du nutriscore est indépendante car elle représente un classement des aliments.

# ### L'égalité des variances

# Réalisons un test de bartlet afin de confirmer ce que nous avons vu lors de l'analyse bivariée.
# 
# H0 : Les variances de chaque groupe sont égales si p-value > 5%
# 
# H1 : Les variances de chaque groupe ne sont pas toutes égales < 5%

# Définissons nos groupes

# In[1696]:


data_a=dataN.loc[data["nutrition_grade_fr"]=="A"]
data_b=dataN.loc[data["nutrition_grade_fr"]=="B"]
data_c=dataN.loc[data["nutrition_grade_fr"]=="C"]
data_d=dataN.loc[data["nutrition_grade_fr"]=="D"]
data_e=dataN.loc[data["nutrition_grade_fr"]=="E"]


# Effectuons le test pour chaque variable quantitative

# In[1697]:


for col in dataN.select_dtypes(include=['float64']).columns:
    print("colonne "+col+" "+str(scipy.stats.bartlett(data_a[col], data_b[col], data_c[col], data_d[col], data_e[col])))


# Nous rejetons h0 pour toutes nos variables. Les variances ne sont pas égales.

# La deuxième condition pour effectuer une ANOVA n'est pas validée.
# 

# In[1698]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols('energy_100g ~ nutrition_grade_fr ', data=dataN).fit()
anova = sm.stats.anova_lm(model, typ=2)

anova


# H0 : les moyennes sont équivalent dans les groupes
# H1 : les moyennes sont différentes
#     
# P<0.05 on rejette H0. et on admet H1.
# 
# #### Les résidus doivent suivre une loi normale

# In[1699]:


import numpy as np
import statsmodels.api as sm
import pylab
from scipy.stats import shapiro
model = ols('energy_100g ~ nutrition_grade_fr', data=dataN).fit()
scipy.stats.normaltest(model.resid)


# ~H0 : Les résidus suivent une loi normale si p-value > 5%~
# H1 : Les résidus ne suivent pas une loi normale si p-value < 5%

# In[1700]:


import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pylab
from scipy.stats import shapiro
model = ols('energy_100g ~ nutrition_grade_fr', data=dataN).fit()


sm.qqplot(model.resid, line='45')
pylab.show()


# ## b) Utilisons le test de Krusdall (utilise les rangs au lieu des moyennes)

# - H0 :  la médiane de la population de tous les groupes est égale pvalue>0.05
# - H1 :  la médiane de la population d'au moins un groupe n'est pas égale p-value<0.05

# Test Krusdall(utilise mediane et quantile)

# In[1701]:


for col in dataN.select_dtypes(include=['float64']).columns:
    print("colonne "+col+": "+ str(scipy.stats.kruskal(*[group[col] for name, group in dataN.groupby("nutrition_grade_fr")])))


# Signatifivement différent on rejette H0 car pvalue<5%

# Selon le test, la mediane d'un groupe sont significativement différents pour chacunes des variables.

# ## Conclusion Nutriscore

#  les aliments avec une note A ?
# - riches en fibres et en proteines 
# - peu de sel, de sucre et de matières grasses (y compris en graisses saturés)
# - contiennent des fruits, végétaux ou des arachides
# - faible en additifs
# - apport en energie proche de la moyenne
# 
#  les aliments avec une note E ?
# - riches en sucre, en additif et en glucides
# - pauvres en fibres, proteines, sel
# - une energie qui varie beaucoup dans les extrêmes.
# 
# 
# Cependant, nous avons besoin d'une alimentation équilibrée. 
# 
# Voici l'apport quotidien pour un adulte : 
# 
# Glucides = 250g (carbohydrates)
# 
# lipides = 50 insaturé et 20 sature
# 
# Protéines =	45 g pour une personne de 55 kg
#             60 g pour une personne de 75 kg
#             
# Fibres =	25 à 30 g dont une moitié issue des céréales et l'autre issue des fruits et légumes
#     
# Donc si nous mangeons que des produits A, cela ne nous permet pas de manger équilibrer et ne permet pas d'améliorer la santé.
# Comment pouvons - nous améliorer la santé des français?

# # III - Objectif une meilleure santé pour tous : Application Smart Food

# Un classement des aliments et le calcul de leur combinaison permettront d'améliorer la santé des français et les guider vers une alimentation équilibrée sans se priver.

# ## 1) Classement des aliments

# Etudions de plus près nos aliments et regardons qu'elles sont les variables qui permettent d'expliquer au mieux cette population.
# 
# 
# Pour commencer, il nous faut exclure les données manquantes et étudier les variables quantitatives qui sont corrélées.

# In[1702]:


data.columns


# In[2911]:


data_quanti=data[['additives_n',
       'ingredients_from_palm_oil_n',
       'ingredients_that_may_be_from_palm_oil_n','energy_100g', 'fat_100g', 'saturated_fat_100g', 'carbohydrates_100g',
       'sugars_100g', 'fiber_100g', 'proteins_100g', 'salt_100g',
       'sodium_100g', 
       'ingredients_from_palm_oil', ]]


# ## a) Corrélations entre les variables quantitatives

# Regardons les variables corrélées.

# In[2917]:


sns.set(rc={'figure.figsize':(10,4)})

data_corr = data_quanti.corr()

display(data_corr)
ax = sns.heatmap(data_corr, xticklabels = data_corr.columns , 
                 yticklabels = data_corr.columns, cmap = 'rocket_r')
plt.title("Matrice de corrélation des variables quantitatives")

plt.xlabel("Variables")

plt.ylabel("Variables")


# Cette matrice nous permet d'identifier les variables corrélées : 
# - fat_100g et saturated_fat_100g
# - carbohydrates_100g et sugars_100g
# - energy_100g et fat_100g
# - salt_100g et sodium_100g
# - additives_n et ingredients_that_may_be_from_palm_oil_n

# ## b) Analyse en composante principale

# Il est necessaire de conserver une des variables corrélées car elles n'apportent aucune information supplémentaires.
# Nous excluons les variables suivantes : 
# - saturated_fat_100g
# - sugars_100g
# - energy_100g
# - sodium_100g
# - ingredients_that_may_be_from_palm_oil_n

# In[2974]:


data_study.columns


# In[2968]:



# suppression des colonnes non numériques
WGI_num0 = data_study.drop(columns =data_study.select_dtypes(include=['object', 'string']).columns)
WGI_num0.columns


# In[2969]:


WGI_num0 = WGI_num0.drop(columns =["reg_additives", "reg_energy", "reg_fat", "reg_carbohydrates", "reg_fiber",
                                  "reg_protein", "nutrition_score_fr_100g",
                                  "energy_100g", "saturated_fat_100g", "sugars_100g", "ingredients_that_may_be_from_palm_oil_n",
                                  "ingredients_from_palm_oil_n"])


# In[2970]:


WGI_num0.columns


# In[2971]:



WGI_num0=WGI_num0.loc[(pd.isna(WGI_num0.carbohydrates_100g)==False) & (pd.isna(WGI_num0.proteins_100g)==False)
                      & (pd.isna(WGI_num0.fat_100g)==False) & (pd.isna(WGI_num0["fiber_100g"])==False) 
                      & (pd.isna(WGI_num0["salt_100g"])==False)
                 & (pd.isna(WGI_num0["additives_n"])==False)]


# In[2972]:


WGI_num0.describe(include="all")


# In[2973]:


data_study.columns


# In[2954]:



#instanciation
sc = StandardScaler()
#transformation – centrage-réduction
Z = sc.fit_transform(WGI_num0)
print(Z)


# In[2955]:


pca = PCA(n_components=6)

#calculs
coord = pca.fit_transform(Z)
pca.fit_transform(Z)


# In[2956]:


print(pca.explained_variance_)
print(pca.explained_variance_ratio_)


# In[2957]:


eig = pd.DataFrame(
    {
        "Dimension" : ["Dim" + str(x + 1) for x in range(6)], 
        "Valeurs propres" : (n-1) / n * pca.explained_variance_,
        "% variance expliquée" : np.round(pca.explained_variance_ratio_ * 100),
        "% cum. var. expliquée" : np.round(np.cumsum(pca.explained_variance_ratio_) * 100)
    }
)
eig


# In[3018]:


eig.plot.bar(x = "Dimension", y = "% cum. var. expliquée", color="g") # permet un diagramme en barres
plt.text(5, 18, "17%") # ajout de texte
plt.axhline(y = 17, linewidth = .5, color = "red", linestyle = "--") # ligne 17 = 100 / 6 (nb dimensions)
plt.show()


# In[3019]:


pca.explained_variance_ratio_


# In[3020]:


#cumul de variance expliquée
plt.plot(np.arange(0,p),np.cumsum(pca.explained_variance_ratio_))
plt.title("Explained variance vs. # of factors")
plt.ylabel("Cumsum explained variance ratio")
plt.xlabel("Factor number")
plt.show()


# On conserve 3 dimensions.

# In[3021]:


# Transformation en DataFrame pandas
WGI_pca_df = pd.DataFrame({
    "Dim1" : WGI_pca[:,0], 
    "Dim2" : WGI_pca[:,1],
    "Dim3" : WGI_pca[:,2],
    "product": data_study["product_name"],
    "nutrition_grade_fr" : data_study["nutrition_grade_fr"]
})

# Résultat (premières lignes)
WGI_pca_df.head()


# In[2962]:


WGI_pca_df.plot.scatter("Dim1", "Dim2") # nuage de points
plt.xlabel("Dimension 1 (24%)") # modification du nom de l'axe X
plt.ylabel("Dimension 2 (21%)") # idem pour axe Y
plt.suptitle("Premier plan factoriel (45%)") # titre général
plt.show()


# In[2963]:


WGI_num


# In[2964]:


n_components


# In[2965]:



# Append the principle components for each entry to the dataframe
for i in range(0, 3):
    data_study['PC' + str(i + 1)] = coord[:, i]

display(WGI_num0.head())

# Show the points in terms of the first two PCs
g = sns.lmplot('PC1',
               'PC2',
               hue='nutrition_grade_fr',data=data_study,
               fit_reg=False,
               scatter=True,
               size=7)

plt.show()


# In[2966]:



# Plot a variable factor map for the first two dimensions.
(fig, ax) = plt.subplots(figsize=(8, 8))
for i in range(0, pca.components_.shape[1]):
    ax.arrow(0,
             0,  # Start the arrow at the origin
             pca.components_[0, i],  #0 for PC1
             pca.components_[1, i],  #1 for PC2
             head_width=0.1,
             head_length=0.1)

    plt.text(pca.components_[0, i] + 0.05,
             pca.components_[1, i] + 0.05,
             WGI_num0.columns.values[i])


an = np.linspace(0, 2 * np.pi, 100)
plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale
plt.axis('equal')
ax.set_title('Variable factor map')
plt.show()


# Le premier plan factoriel est expliqué par les variables proteins et fat positivement
# Et le second plan factoriel est expliqué par les variables fibres et carbohydrate. Cependant, ces variables sont assez éloignées du bord du cercle.
# 
# Nous devons calculer le cosinus au carré pour vérifier les coefficients de corrélation avec nos axes.
# 
# Combinons nos deux graphiques précédent.

# In[2929]:


cos2var = corvar**2
print(pd.DataFrame({'id':WGI_num0.columns,'COS2_1':cos2var[:,0],'COS2_2':cos2var[:,1],'COS2_3':cos2var[:,2]}))


# In[ ]:


#cosinus carré des variables
cos2var = corvar**2
print(pandas.DataFrame({'id':WGI_num0.columns,'COS2_1':cos2var[:,0],'COS2_2':cos2var[:,1],'COS2_3':cos2var[:,1]}))


# In[2931]:


#contributions
ctrvar = cos2var
for k in range(p):
 ctrvar[:,k] = ctrvar[:,k]/eigval[k]
#on n'affiche que pour les trois premiers axes
print(pd.DataFrame({'id':WGI_num0.columns,'CTR_1':ctrvar[:,0],'CTR_2':ctrvar[:,1],'CTR_3':ctrvar[:,2]}))


# In[1897]:


####mettre en 3d
from yellowbrick.datasets import load_concrete
from yellowbrick.features import PCA
from yellowbrick.style import set_palette

# Load the concrete dataset
plt.figure()
visualizer = PCA(scale=True, proj_features=True)
visualizer.fit_transform(WGI_num0)
visualizer.poof()


# Nous observons 2 groupes extrêmes :un qui est caractérisé par un fort nombre de carbohydrates et de fibres et le deuxième groupe qui est caractérisé par un fort nombre de protéines et de matières grasses.
# 
# Des variables semblent être liées : 
# - carbohydrates et fibres
# - fat et protéine
# 
# Réalisons une classification hiérarchique afin de classer nos aliments.
# 

# ## c ) Classification ascendante hiérarchique

# Nous allons extraire un échantillon de notre base de données afin de réduire la population. Les calculs sur une population importante pose problème selon la puissance de l'ordinateur.

# In[2655]:


data_study["nutrition_grade_fr"].value_counts()


# In[2656]:


test_a=data_study.loc[data_study["nutrition_grade_fr"]=="A"].sample(1100)
test_b=data_study.loc[data_study["nutrition_grade_fr"]=="B"].sample(1100)
test_c=data_study.loc[data_study["nutrition_grade_fr"]=="C"].sample(1100)
test_d=data_study.loc[data_study["nutrition_grade_fr"]=="D"].sample(1100)
test_e=data_study.loc[data_study["nutrition_grade_fr"]=="E"].sample(1047)


# In[2657]:


frames = [test_a, test_b, test_c, test_d, test_e]


# In[2658]:


test = pd.concat(frames)


# In[2659]:


test.shape


# In[2660]:


test2=test.drop(columns =data_study.select_dtypes(include=['object', 'string']).columns)


# In[2661]:


test3 = test2.drop(columns =["reg_additives", "reg_energy", "reg_fat","reg_carbohydrates", "reg_fiber",
                                  "reg_protein","ingredients_from_palm_oil_n",
                             "ingredients_that_may_be_from_palm_oil_n",
                                  "energy_100g", "saturated_fat_100g",  "sugars_100g"])


# In[2662]:


test2 = test2.drop(columns =["reg_additives", "reg_energy", "reg_fat","reg_carbohydrates", "reg_fiber",
                                  "reg_protein", "nutrition_score_fr_100g","ingredients_from_palm_oil_n",
                             "ingredients_that_may_be_from_palm_oil_n",
                                  "energy_100g", "saturated_fat_100g",  "sugars_100g"])


# In[2919]:


test3.columns


# In[2663]:


test2.describe(include="all")


# ### Réalisons un dendrogramme

# In[2922]:


# Centrage et Réduction
std_scale = preprocessing.StandardScaler().fit(test2)
X_scaled = std_scale.transform(test2)


# In[2923]:


#librairies pour la CAH
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

#générer la matrice des liens
Z = linkage(X_scaled,method='ward',metric='euclidean')
#affichage du dendrogramme
plt.title("CAH")
dendrogram(Z,labels=test2.index,color_threshold=0)
plt.show()


# Nous pourrions faire 5 groupes. Réalisons ce découpage en affichant le nutriscore.

# In[1368]:


#matérialisation des 5 classes (hauteur t = 350)
plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index or (cluster size)')
plt.ylabel('distance')
dendrogram(Z,labels=test.index,orientation='left',color_threshold=350,
           truncate_mode = 'lastp' ,   # afficher uniquement les p derniers clusters fusionnés 
    p = 12 ,   # afficher uniquement les p derniers clusters fusionnés 
   
    leaf_font_size = 12. , 
    show_contracted = True
          )
plt.show()


# In[1437]:


#découpage à la hauteur t = 350 ==> identifiants de 5 groupes obtenus
groupes_cah = fcluster(Z,t=350,criterion='distance')
print(groupes_cah)
#index triés des groupes
idg = np.argsort(groupes_cah)
#affichage des observatbbions et leurs groupes
info_groupe=pd.DataFrame(test.index[idg],groupes_cah[idg])


# In[1439]:


WGI_pca_k2 = test.assign(classe = info_groupe.index)


# In[1441]:


t = pd.crosstab(WGI_pca_k2.nutrition_grade_fr, WGI_pca_k2.classe, normalize = "columns")
t = t.assign(nutrition_grade_fr = t.index)
tm = pd.melt(t, id_vars = "nutrition_grade_fr")
tm = tm.assign(value = 100 * tm.value)

sns.catplot("nutrition_grade_fr", y = "value", col = "classe", data = tm, kind = "bar")


# ## d) Essayons d'utiliser KMeans

# In[769]:


from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from yellowbrick.cluster import KElbowVisualizer

X = WGI_num0
# Instantiate the clustering model and visualizer
model = KMeans()
visualizer = KElbowVisualizer(
    model, k=(4,12), metric='calinski_harabasz', timings=False, locate_elbow=False
)

visualizer.fit(X)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure


# In[ ]:


from yellowbrick.cluster import KElbowVisualizer

# Generate synthetic dataset with 8 random clusters
X = WGI_num0

# Instantiate the clustering model and visualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(4,11))

visualizer.fit(X)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure


# Nous essayerons 5 groupes.

# In[203]:


from sklearn.cluster import KMeans

kmeans2 = KMeans(n_clusters = 5)
kmeans2.fit(scale(WGI_num0))


# In[204]:


kmeans2.labels_


# In[205]:


pd.Series(kmeans2.labels_).value_counts()


# In[206]:


kmeans2.cluster_centers_


# In[207]:


WGI_k2 = WGI_num0.assign(classe = kmeans2.labels_)
WGI_k2.groupby("classe").mean()


# In[208]:


WGI_k2


# In[209]:


WGI_pca_k2 = WGI_pca_df.assign(classe = kmeans2.labels_)
WGI_pca_k2.plot.scatter(x = "Dim1", y = "Dim2", c = "classe", cmap = "Accent")
plt.show()


# In[210]:


WGI_pca_k2.shape


# In[211]:


ax = plt.figure(figsize=(16,10)).gca(projection='3d')
WGI_pca_k2 = WGI_pca_df.assign(classe = kmeans2.labels_)
ax.scatter(
    xs=WGI_pca_k2["Dim1"], 
    ys=WGI_pca_k2["Dim2"], 
    zs=WGI_pca_k2["Dim3"],
    c=WGI_pca_k2["classe"],
    cmap='tab10'
)
ax.set_xlabel('Dim 1 (35%)')
ax.set_ylabel('Dim 2(30%)')
ax.set_zlabel('Dim 3 (21%)')
plt.show()


# In[212]:


inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters = k, init = "random", n_init = 20).fit(WGI_num0)
    inertia = inertia + [kmeans.inertia_]
inertia = pd.DataFrame({"k": range(1, 11), "inertia": inertia})
inertia.plot.line(x = "k", y = "inertia")
plt.scatter(2, inertia.query('k == 2')["inertia"], c = "red")
plt.scatter(4, inertia.query('k == 4')["inertia"], c = "red")
plt.show()


# coude plus marqué au niveau du 4

# In[213]:


WGI_pca_k2


# Ces 4 classes vont permettre de manger équilibrer en les combinant.

# Comparons les nouveaux groupes et le nutriscore.

# In[214]:


pd.crosstab(WGI_pca_k2.classe, WGI_pca_k2.nutrition_grade_fr, normalize = True)


# In[215]:


sns.heatmap(pd.crosstab(WGI_pca_k2.nutrition_grade_fr, WGI_pca_k2.classe, normalize = True))


# In[216]:


t = pd.crosstab(WGI_pca_k2.nutrition_grade_fr, WGI_pca_k2.classe, normalize = "columns")
t = t.assign(nutrition_grade_fr = t.index)
tm = pd.melt(t, id_vars = "nutrition_grade_fr")
tm = tm.assign(value = 100 * tm.value)

sns.catplot("nutrition_grade_fr", y = "value", col = "classe", data = tm, kind = "bar")


# Aucun groupe ne semble se demarquer, excepté la classe 2.
# 
# Les autres groupes sont homogènes. Ce découpage ne semble pas adapter et est assez aléatoire.

# ## e) Testons un mix de Kmeans et CAH

# Créons un nombre de groupe important avec kmeans

# In[2224]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale
# suppression des co!lonnes non numériques
WGI_num0 = data_study.drop(columns =data_study.select_dtypes(include=['object', 'string']).columns)
WGI_num0 = WGI_num0.drop(columns =["reg_additives", "reg_energy", "reg_fat", "reg_carbohydrates", "reg_fiber",
                                  "reg_protein", "nutrition_score_fr_100g",
                                  "energy_100g", "saturated_fat_100g", "sugars_100g", "ingredients_that_may_be_from_palm_oil_n",
                                  "ingredients_from_palm_oil_n"])
WGI_num0=WGI_num0.loc[ (pd.isna(data.carbohydrates_100g)==False) & (pd.isna(data.proteins_100g)==False) 
                       & (pd.isna(data.fat_100g)==False)
       & (pd.isna(data["fiber_100g"])==False)
                       & (pd.isna(data["salt_100g"])==False)
& (pd.isna(data["nutrition_score_fr_100g"])==False)
                 & (pd.isna(data["nutrition_grade_fr"])==False)
                 & (pd.isna(data["additives_n"])==False)]


# In[2225]:


WGI_num0.columns


# In[2226]:


WGI_num0.shape


# In[2723]:


from sklearn.cluster import KMeans

kmeans2 = KMeans(n_clusters = 50)
kmeans2.fit(scale(WGI_num0))


# In[2728]:


labels=kmeans2.labels_


# In[2729]:


labels.shape


# In[2730]:


centroid=kmeans2.cluster_centers_


# In[2731]:



#perform the clustering
Z = linkage(centroid,method='ward',metric='euclidean')


# In[2735]:



#affichage du dendrogramme
plt.title("CAH")
dendrogram(Z,color_threshold=11)
plt.show()


# Réalisons la CAH sur le centre des clusters

# In[2736]:


#découpage à la hauteur t = 400 ==> identifiants de 5 groupes obtenus
groupes_cah = fcluster(Z,t=11,criterion='distance')
print(groupes_cah)


# In[2737]:


#index triés des groupes
idg = np.argsort(groupes_cah)
#affichage des observatbbions et leurs groupes
info_groupe=pd.DataFrame(centroid[idg],groupes_cah[idg], columns=['0','1','2','3','4','5'])


# In[2738]:


info_groupe


# Faisons une jointure entre kmeans et cah

# In[2739]:


info_groupe


# In[2740]:


cluster_map = pd.DataFrame()
cluster_map['data_index'] = WGI_num0.index
cluster_map['cluster'] = kmeans2.labels_


# In[2741]:


max(cluster_map["cluster"])


# In[2742]:


centroid_map = pd.DataFrame(columns=["cluster", "0", "1", "2", "3", "4", "5"])
for i in range(0,max(cluster_map["cluster"])+1):
    centroid_map.loc[i] = i, kmeans2.cluster_centers_[i][0],kmeans2.cluster_centers_[i][1], kmeans2.cluster_centers_[i][2], kmeans2.cluster_centers_[i][3],kmeans2.cluster_centers_[i][4],kmeans2.cluster_centers_[i][5]
    


# In[2743]:


centroid_map.shape


# In[2744]:


info_groupe["classe_cah"]=info_groupe.index


# In[2745]:


cluster_map


# In[2746]:


data_merge = pd.merge(centroid_map, info_groupe, how="left", on=["0", "1", "2","3","4","5"], indicator=True,  suffixes=('', '_del'))
data_result = data_merge.loc[data_merge["_merge"] == "both"].drop("_merge", axis=1)


# In[2747]:


cah=data_result


# In[2748]:


cluster_map.describe()


# In[2749]:


data_merge = pd.merge(cluster_map, cah, how="left", on=["cluster"], indicator=True,  suffixes=('', '_del'))
data_result = data_merge.loc[data_merge["_merge"] == "both"].drop("_merge", axis=1)



# In[2750]:


mix_kmeans_cah=data_result


# In[2751]:


mix_kmeans_cah


# In[2752]:


data_study


# In[2753]:



WGI_k2 = data_study.assign(classe = mix_kmeans_cah.classe_cah)
WGI_k2.groupby("classe").mean()


# In[2754]:


t = pd.crosstab(WGI_k2.nutrition_grade_fr, WGI_k2.classe, normalize = "columns")
t = t.assign(nutrition_grade_fr = t.index)
tm = pd.melt(t, id_vars = "nutrition_grade_fr")
tm = tm.assign(value = 100 * tm.value)

sns.catplot("nutrition_grade_fr", y = "value", col = "classe", data = tm, kind = "bar")


# ## f) CAH : Utilisation des variables quantitatives avec regroupement

# Essayons de refaire les calculs avec nos variables "regroupement"

# Pour rappel :
# - Points négatifs : l'énergie, les graisses saturées, les sucres, et le sel (des niveaux élevés sont considérés comme mauvais pour la santé)
# 
# - Points positifs : la proportion de fruits, de légumes, de noix, d'huiles d'olive, de colza et de noix, de fibres et de protéines (les niveaux élevés sont considérés comme bons pour la santé).

# In[212]:


data_study = data[["code", "product_name",  #"score_total",
            "ingredients_text","energy_100g", "carbohydrates_100g", "proteins_100g", "fat_100g",
            "saturated_fat_100g", "fiber_100g", "sugars_100g", "salt_100g",
             "additives_n", "reg_salt",
            "ingredients_from_palm_oil_n", "ingredients_that_may_be_from_palm_oil_n", 
            "nutrition_grade_fr", "main_category", "nutrition_score_fr_100g", 'reg_additives',
       'reg_energy', 'reg_fat', 'reg_carbohydrates', 'reg_fiber', 
       'reg_protein']]
data_study=data_study.loc[(pd.isna(data_study.reg_additives)==False) 
               & (pd.isna(data_study.reg_carbohydrates)==False) & (pd.isna(data_study.reg_fat)==False) 
               & (pd.isna(data_study.reg_fiber)==False)
                            & (pd.isna(data_study.nutrition_score_fr_100g)==False)
                           & (pd.isna(data_study.nutrition_grade_fr)==False)
                          & (pd.isna(data_study.reg_protein)==False)
                          & (pd.isna(data_study.reg_salt)==False)]


# In[214]:



WGI_num_t2 = data_study.drop(columns = ["nutrition_score_fr_100g", "energy_100g","main_category", 
                                      "product_name","sugars_100g", "saturated_fat_100g","code", 
                                      "ingredients_text","reg_energy","salt_100g",
                                     "carbohydrates_100g", "proteins_100g", "fat_100g",
                                     "fiber_100g",  "additives_n", "ingredients_from_palm_oil_n", "ingredients_that_may_be_from_palm_oil_n"])

WGI_num_t2.columns


# In[215]:


WGI_num_t2["nutrition_grade_fr"].value_counts()


# In[216]:


test_a=WGI_num_t2.loc[WGI_num_t2["nutrition_grade_fr"]=="A"].sample(2000)
test_b=WGI_num_t2.loc[WGI_num_t2["nutrition_grade_fr"]=="B"].sample(2000)
test_c=WGI_num_t2.loc[WGI_num_t2["nutrition_grade_fr"]=="C"].sample(2000)
test_d=WGI_num_t2.loc[WGI_num_t2["nutrition_grade_fr"]=="D"].sample(2000)
test_e=WGI_num_t2.loc[WGI_num_t2["nutrition_grade_fr"]=="E"].sample(1047)

frames = [test_a, test_b, test_c, test_d, test_e]

test = pd.concat(frames)


# In[217]:


test.columns


# In[218]:


test2=test.copy()
del test2["nutrition_grade_fr"]


# In[220]:



#librairies pour la CAH
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
#générer la matrice des liens
Z = linkage(test2,method='ward',metric='euclidean')
#affichage du dendrogramme
plt.title("CAH")
dendrogram(Z,labels=test.index,color_threshold=60)
plt.show()


# In[221]:


#matérialisation des 5 classes (hauteur t = 350)
plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index or (cluster size)')
plt.ylabel('distance')
dendrogram(Z,labels=test2.index,orientation='left',color_threshold=60,
           truncate_mode = 'lastp' ,   # afficher uniquement les p derniers clusters fusionnés 
    p = 12 ,   # afficher uniquement les p derniers clusters fusionnés 
   
    leaf_font_size = 12. , 
    show_contracted = True
          )
plt.show()


# In[222]:


#découpage à la hauteur t = 400 ==> identifiants de 5 groupes obtenus
groupes_cah = fcluster(Z,t=60,criterion='distance')
print(groupes_cah)
#index triés des groupes
idg = np.argsort(groupes_cah)
#affichage des observatbbions et leurs groupes
info_groupe=pd.DataFrame(test.index[idg],groupes_cah[idg])


# In[223]:


WGI_pca_k2 = test.assign(classe = info_groupe.index)


# In[224]:


t = pd.crosstab(WGI_pca_k2.nutrition_grade_fr, WGI_pca_k2.classe, normalize = "columns")
t = t.assign(nutrition_grade_fr = t.index)
tm = pd.melt(t, id_vars = "nutrition_grade_fr")
tm = tm.assign(value = 100 * tm.value)

sns.catplot("nutrition_grade_fr", y = "value", col = "classe", data = tm, kind = "bar")


# Ces résultats sont assez proche du nutriscore. Il y a des poids à mettre en fonction des apports nutritionnels.
# Il faut donc appliquer une règle métier pour séparer les groupes qui sont proches A/B et DE.

# # IV - Combinaison des aliments

# Utilisons le nutriscore qui permettra de classer nos aliments. Nous devons modifier les étiquettes et créer des catégories d'aliment afin de pouvoir garantir une alimentation équilibrée.
# 
# Ensuite, nous devons comparer le menu quotidien avec le seuil de l’apport journalier.
# 
# Voici l'apport quotidien pour un adulte :
# Glucides = 250g (carbohydrates)
# lipides = 50 insaturé et 20 sature
# Protéines = 45 g pour une personne de 55 kg 60 g pour une personne de 75 kg
# Fibres = 25 à 30 g dont une moitié issue des céréales et l'autre issue des fruits et légumes
# 
# Nous pourrions créer une application Mobile Smart-Food.
# 

# # Conclusion - Un Apport journalier équilibré

# Le nutriscore peut permettre d’améliorer la santé des français. En effet, il permet de catégoriser les aliments selon l’influence des facteurs suivants : matières grasses, additifs, fibres, protéines, glucides, sels. 
# Il faut aussi ajouter des poids métiers afin de noter les aliments positivement ou négativement selon ces facteurs.
# 
# Cependant, c’est la combinaison de ces aliments qui va permettre aux français de manger équilibrer et de recevoir un apport nutritionnel quotidien suffisant.
# 
# Ainsi, il est nécessaire de proposer une application – Smart-food et de modifier les « étiquettes » du nutriscore afin de catégoriser les aliments et que ce soit plus parlant pour la population.
# 

# # Pour aller plus loin : Knn classifier

# Remplaçons les données manquantes du nutrition grade fr. Utilisons le Knn classifier qui permet de prédire les données selon les plus proche voisins.

# Utilisons les variables regroupées que nous avons utilisé pour la CAH.

# In[240]:


data_knn = data[['reg_salt', 'nutrition_grade_fr', 'reg_additives', 'reg_fat',
       'reg_carbohydrates', 'reg_fiber', 'reg_protein']]


# In[241]:


data_knn = data_knn.loc[(pd.isna(data_knn["reg_salt"])==False) & (pd.isna(data_knn["reg_additives"])==False) 
                        & (pd.isna(data_knn["reg_fat"])==False) & (pd.isna(data_knn["reg_carbohydrates"])==False) 
                        & (pd.isna(data_knn["reg_fiber"])==False) & (pd.isna(data_knn["reg_protein"])==False)]


# In[248]:


data_knn_train = data_knn.loc[pd.isna(data_knn["nutrition_grade_fr"])==False]


# In[249]:


data_knn_train.shape


# In[273]:


#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
Nutgrade_encoded=le.fit_transform(data_knn_train["nutrition_grade_fr"])
print(Nutgrade_encoded)


# In[274]:


features=list(zip(data_knn_train["reg_salt"], data_knn_train["reg_additives"], data_knn_train["reg_fat"]
                 , data_knn_train["reg_carbohydrates"], data_knn_train["reg_fiber"], data_knn_train["reg_protein"]))


# In[275]:


# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(features, Nutgrade_encoded, test_size=0.3) # 70% training and 30% test


# In[276]:


#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)


# In[277]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# Taux de classification à 64,6%

# #### Appliquons ce modèle sur les données à prédire

# In[250]:


data_knn_test = data_knn.loc[pd.isna(data_knn["nutrition_grade_fr"])==True]


# In[251]:


data_knn_test.shape


# In[252]:


data_knn_test.columns


# Commençons par encoder le nutrion grade fr

# In[263]:


features_test=list(zip(data_knn_test["reg_salt"], data_knn_test["reg_additives"], data_knn_test["reg_fat"]
                 , data_knn_test["reg_carbohydrates"], data_knn_test["reg_fiber"], data_knn_test["reg_protein"]))


# Construisons le modèle de classificateur KNN.

# In[261]:


from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)


# In[262]:


# Train the model using the training sets
model.fit(features,Nutgrade_encoded)


# In[265]:


predicted= model.predict(features_test) # 0:Overcast, 2:Mild


# In[268]:


data_knn_test=data_knn_test.assign(nutrition_grade_fr=predicted)


# In[278]:


pk.graph_barplot(data_knn_test['nutrition_grade_fr'], "Répartition des aliments selon le nutrition grade prédit avec Knn classifier", 
              (0.82, 0.28, 0.09),
              0, 45, "nutrition_grade_fr", "Fréquence en %",0,1, (11,7))

