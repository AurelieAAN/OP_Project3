#!/usr/bin/env python
# coding: utf-8

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

# ## Conclusion qualitatives

# Nous avons appris que les variables qualitatives étaient très disperséesau sein des catégories et que la majorité des aliments étaient dans la catégorie "AUTRES".
# 
# Nous avons aussi remarqué que nos données ont été créées en mars 2017 et que les aliments proviennent des USA.
# Nous savons aussi que nos aliments sont bien notés, 32% des aliments ont un B et 28% des aliments ont un A contre moins de 5% des aliments ont un E.

# Maintenant que nous connaissons nos données. Il est important de pouvoir faire un zoom sur le nutriscore afin d'améliorer la santé des français.
