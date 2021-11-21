---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: venv
  language: python
  name: venv
---

# Contexte du projet

+++

## Problématique : Comment améliorer la santé des français ?

+++

## But de l’étude :
- Est-ce que le nutriscore y contribue ?
- Si non, comment pouvons -  nous améliorer la santé de la population ?

+++

## Methode :
- Découvrir les caractéristiques du nutriscore et son intérêt
- Etudier les autres moyens à notre disposition pour améliorer l’efficacité du nutriscore ou apporter de nouvelles solutions

+++

# Construction de la base de données

+++

## Exploration  de nos données

```{code-cell} ipython3
import pandas as pd
import numpy as np
import missingno
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats
import seaborn as sns
from Package import Scripts_Analyse01 as pk
import plotly.express as px
import plotly.graph_objects as go
from yellowbrick.features import ParallelCoordinates
from plotly.graph_objects import Layout
```

```{code-cell} ipython3
data = pd.read_csv("./Data/fr.openfoodfacts.org.products.csv", encoding='utf-8', sep="\t",low_memory=False)
```

```{code-cell} ipython3
data.shape
```

Nous disposons de 320 772 produits alimentaires et de 162 colonnes.

+++

Regardons le nom des colonnes.

```{code-cell} ipython3
for i in data.columns:
    print(i)
```

Nous disposons du détail des aliments en fonction notamment des apports nutritionnels.

Regardons quel type de variables nous avons.

```{code-cell} ipython3
data.info()
```

Essayons de transformer les variables object en string.

```{code-cell} ipython3
#pk.data_transformcol_string(data)
```

Etudions un extrait de nos donnéees.

```{code-cell} ipython3
data.head(5)
```

Nous observons beaucoup de données manquantes. Regardons les variables avec plus de 80 %  de données manquantes.

```{code-cell} ipython3
tab=pk.del_Nan(data, 0.8,0, 0)
```

```{code-cell} ipython3
pd.set_option('display.max_rows', 500)
tab

```

Ces variables n'apportent pas suffisament d'information. Nous pouvons les supprimer.

```{code-cell} ipython3
pk.del_Nan(data, 0.8,1, 2)
```

```{code-cell} ipython3
data.shape
```

Il nous reste 54 variables.

+++

Regardons si des variables ne sont pas utiles.

```{code-cell} ipython3
data.columns
```

Certaines variables ne sont pas utiles pour notre analyse comme l'url des fiches ou la date de dernière modification.
Nous pouvons donc les supprimer et conserver par exemple la date de création de la fiche aliment.
Voici les colonnes que nous supprimons : 
url, creator, created_t, image_url, image_small_url, last_modified_t, last_modified_datetime

```{code-cell} ipython3
data.drop(["url", "creator", "created_t","image_url","image_small_url", "last_modified_t", "last_modified_datetime"], axis=1, inplace=True)
```

Ensuite, nous pouvons supprimer les variables avec des données uniques.

```{code-cell} ipython3
data=pk.data_uniqueone_string(data)
```

```{code-cell} ipython3
data.shape
```

A présent, nous pouvons tracer la matrice des données manquantes pour mieux observer ces valeurs.

```{code-cell} ipython3
pk.matrix_vm(data, (16,8), (0.60, 0.64, 0.49))
```

Il reste beaucoup de données manquantes. Mais nous pouvons créer des catégories N.A pour les variables qualitatives

```{code-cell} ipython3
pk.data_fillNA_string(data, "AUTRES")
```

```{code-cell} ipython3
pk.matrix_vm(data, (14,8), (0.82, 0.28, 0.09))
```

Il semble que des colonnes avec plus de 50% de données manquantes soient encore présentes.
Regardons de plus près ces colonnes.

```{code-cell} ipython3
missing_value_df = pk.data_missingTab(data)
```

```{code-cell} ipython3
def graph_int_bar(data_i, x_i, y_i, x_label_i, y_label_i, palette_color_i, title_i):
    
    fig = px.bar(ms, x=x_i, y=y_i, color=y_i, 
             labels={x_i:x_label_i,y_i:y_label_i},
              color_continuous_scale=palette_color_i)
    fig.update_layout(
        title_text=title_i, # title of plot
    )
    fig.show()
```

```{code-cell} ipython3
ms =missing_value_df.loc[missing_value_df["percent_missing"]>50]
graph_int_bar(ms, 'column_name', 'percent_missing', "Variables", "% - Valeurs manquantes", "fall", "Variables avec plus de 50% de valeurs manquantes")
```

Nous décidons de supprimer ces colonnes.

```{code-cell} ipython3
pk.del_Nan(data, 0.5,1, 2)
```

Regardons les données manquantes restantes.

```{code-cell} ipython3
missing_value_df = pk.data_missingTab(data)
ms =missing_value_df.loc[missing_value_df["percent_missing"]>0]
graph_int_bar(ms, 'column_name', 'percent_missing', "Variables", "% - Valeurs manquantes", "fall", "Répartition des variables en fonction de leur pourcentage de valeurs manquantes")
```

Il reste peu de données manquantes. 

Nous conservons nos données telle quelle pour ne pas perdre d'information.

+++

Regardons quelques statistiques descriptives sur nos variables restantes.

```{code-cell} ipython3
data.describe()
```

```{code-cell} ipython3
data[data.select_dtypes(include=['object', 'string']).columns].describe()
```

Renommons les variables qui contiennent des tirets.

```{code-cell} ipython3
data = data.rename(columns={'saturated-fat_100g': 'saturated_fat_100g', 
                       'nutrition-score-fr_100g': 'nutrition_score_fr_100g'})
```

```{code-cell} ipython3
data.shape
```

Nous avons conservé 41 colonnes.
Passons au nettoyage de nos données. 

+++

## Nettoyage des données

+++

### Détection des doublons

+++

Vérifions s'ils existent des doublons (toutes les colonnes identiques), mettons d'abord toutes les catégories en majuscule.

```{code-cell} ipython3
data=pk.data_majuscule(data)
```

```{code-cell} ipython3
data.duplicated().sum()
```

On considère comme produit identique les lignes qui ont le même nom de produit et le même code barre.
En effet, il est noté que des articles différents peuvent avoir le même code barre.

```{code-cell} ipython3
data[['code', 'product_name']].duplicated().sum()
```

Regardons ces doublons

```{code-cell} ipython3
data_doublons = data.loc[data[['code', 'product_name']].duplicated(keep=False),:]
```

```{code-cell} ipython3
data_doublons
```

Ces produits ne sont pas forcément des doublons car ils ne contiennent pas de code identifiant les produits.

Cependant, Le nom  des produits semblent correspondre au pays de provenance et ces lignes ne contiennent pas beaucoup d'informations. Nous observons beaucoup de valeurs manquantes.

Nous décidons de supprimer ces lignes afin de ne pas fausser nos résultats car il semble que la saisie de ces produits ait été mal réalisée.

+++

Faisons une jointure entre notre dataframe de base et notre dataframe qui contient les lignes mal renseignées.

```{code-cell} ipython3
data_result = pd.merge(data, data_doublons, on=["code", "product_name"], how="outer", indicator=True,  suffixes=('', '_del') )
```

Maintenant que nous avons identifié ces lignes dans le dataframe de base, on peut les supprimer.

```{code-cell} ipython3
data_result = data_result.loc[data_result["_merge"] == "left_only"].drop("_merge", axis=1)
```

Conservons les colonnes de notre dataframe de base.

```{code-cell} ipython3
data_result = data_result[[c for c in data_result.columns if not c.endswith('_del')]]
```

```{code-cell} ipython3
data=data_result
```

```{code-cell} ipython3
data[['code', 'product_name']].duplicated().sum()
```

```{code-cell} ipython3
data.shape
```

Création d'une colonne présence possible d'huile de palme : 
Nous vérifions s'il y a de l'huile de palme possible et s'il y en a dans les ingredients.

```{code-cell} ipython3
data.loc[data.ingredients_text.str.contains('PALM')==True  , "ingredients_from_palm_oil"]="OUI"
data.loc[data.ingredients_from_palm_oil_n>=1, "ingredients_from_palm_oil" ]="OUI"
data.loc[data.ingredients_that_may_be_from_palm_oil_n>=2, "ingredients_from_palm_oil" ]="OUI"
data['ingredients_from_palm_oil'].fillna("NON", inplace=True)
```

```{code-cell} ipython3
data.info()
```

```{code-cell} ipython3
pk.matrix_vm(data, (14,8), (0.43,0.51,0.38))
```

Supprimons les données qui n'ont aucune colonne quantitatives renseignées (fin _100g).
En effet, ces colonnes nous permettront de réaliser notre analyse donc il nous faut au moins une information remplie pour une donnée.

```{code-cell} ipython3
data=data.loc[(pd.isna(data.energy_100g)==False) | (pd.isna(data.carbohydrates_100g)==False) 
              | (pd.isna(data.proteins_100g)==False) | (pd.isna(data.fat_100g)==False)
        | (pd.isna(data["saturated_fat_100g"])==False) | (pd.isna(data["fiber_100g"])==False)
         | (pd.isna(data["sugars_100g"])==False)
        | (pd.isna(data["salt_100g"])==False)
             | (pd.isna(data["sodium_100g"])==False)]
```

```{code-cell} ipython3
data.shape
```

Il reste encore certaines données manquantes, mais les individus apportent suffisamment d'informations pour les conserver. 

Passons à la prochaine étape : Etudions chacune de nos variables.

+++

# Exploration des données

+++

### Variables quantitatives

```{code-cell} ipython3
data.describe().round(2)
```

il semble qu'il y a ait des données aberrantes. Nous avons des données négatives pour certaines variables.
Et nous avons des maximums étonnant comparés au 3ème quartile. Même si l'écart-type n'est pas très important.

+++

#### Etudes des valeurs aberrantes

+++

Traçons des stripplots pour toutes nos variables quantitatives. Ces graphiques permettent plus facilement d'observer les valeurs extrêmes pour de grands volumes de données.

```{code-cell} ipython3
for col in data.select_dtypes(include=['float64']).columns:
    pk.graph_stripplot(data,col, "Nuage de point de la variable "+col,(5,3),"#6D8260")
```

Nous observons des données extrêmes dites données aberrantes pour presque toutes nos variables.

Pour rappel, les valeurs aberrantes sont des données qui ne sont pas anodines pour le jeu de données dont on dispose.

Commençons par essayer de supprimer ces données grâce aux informations sur internet.
Nous découvrons que l'aliment le plus calorique contient 900 calories. Nous pouvons donc supprimer les données supérieures ou égales à 901 calories.

source : https://sante.journaldesfemmes.fr/calories/classement/aliments/calories (provenant de l'Anses)

```{code-cell} ipython3
data=pk.delete_outliers_UPPER(data, data['energy_100g'], 901)
```

Pour l'instant pour chacunes de ces colonnes, nous pouvons supprimer toutes les données supérieures à la portion de 100g et inférieures à 0.

+++

Nous pourrions calculer l'écart-interquartile, mais celui-ci ne serait pas adapté à ces données. En effet certains aliments peuvent être très proche de 100g et d'autres proche de 0.

Calculons quand même un exemple pour la variable energy_100g

```{code-cell} ipython3
pk.outliers(data, data['energy_100g'],0)
```

Nous obtenons un écart-interquartile de 1 148 pour la valeur haute (donc largement au-dessus de 901 calories).
Et nous obtenons une valeur inférieur à 0 pour les valeurs basses. Ce qui n'est pas possible pour les nutritions d'un aliment.

Ce calcul n'est donc pas adaptées à nos données car elles sont irréelles par rapport aux aliments.

+++

Supprimons les données supérieures à 100g et inférieurs ou égales à 0 pour les autres "nutriments".

```{code-cell} ipython3

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
```

Retraçons nos stripplot.

```{code-cell} ipython3
for col in data.select_dtypes(include=['float64']).columns:
    pk.graph_stripplot(data,col, "Nuage de point de la variable "+col,(5,3), (0.82, 0.28, 0.09))
```

Regardons plus précisément les variables proteins_100g, saturated-fat_100g, fiber_1100g, il semble qu'il reste des valeurs aberrantes.

+++

#### Commençons par la variable proteins_100g

```{code-cell} ipython3
def boxplot_int(data_var_i,y_label_i, palette_color_i , mean_i):
    layout = Layout(plot_bgcolor='rgba(0,0,0,0)')

    # Use that layout here
    fig = go.Figure(layout=layout)

    fig.update_xaxes(showline=True, linewidth=2, linecolor='#E3E3E3', gridcolor='#E3E3E3', mirror=True )
    fig.update_yaxes(showline=True, linewidth=2, linecolor='#E3E3E3', gridcolor='#E3E3E3', mirror=True)

    if mean_i==1:
        fig.add_trace(go.Box(y=data_var_i,name=y_label_i, 
                        marker_color = palette_color_i, boxmean='sd'))
    else:
        fig.add_trace(go.Box(y=data_var_i,name=y_label_i, 
                        marker_color = palette_color_i))
    
    fig.update_layout(
        title_text="Boite à moustache de la variable "+y_label_i, # title of plot
    )
    

    
    fig.show()
```

```{code-cell} ipython3
boxplot_int(data.proteins_100g, "Protéine", "indianred",0 )
```

L'aliment le plus protéiné est : gélatine alimentaire	87,6 g


source : https://sante.journaldesfemmes.fr/calories/classement/aliments/proteines

+++

Supprimons les données qui ont des protéines supérieures à 89g.

```{code-cell} ipython3
data = data.loc[data["proteins_100g"]<=89]
```

Regardons à nouveau la boxenplot de la variable proteins_100g.

```{code-cell} ipython3
boxplot_int(data.proteins_100g, "Protéine", "indianred",1 )
```

Il reste quelques valeurs extrêmes mais elles sont cohérentes, donc on les conserve pour ne pas perdre d'information.

Les données ne sont pas très dispersées la boite à moustache est aplatie.

+++

#### Regardons à présent la variable saturated-fat_100g

+++

Nous savons que l'aliment qui contient le plus d'acide gras saturés est : pain de friture	92,6 g

+++

source: https://sante.journaldesfemmes.fr/calories/classement/aliments/acides-gras-satures

```{code-cell} ipython3
data = data.loc[data["saturated_fat_100g"]<=93]
```

```{code-cell} ipython3
def graph_int_violin(data_var_i, fillcolor_i, x_label_i):
    layout = Layout(plot_bgcolor='rgba(0,0,0,0)')

    # Use that layout here
    fig = go.Figure(data=go.Violin(y=data_var_i, box_visible=True, line_color='black',
                               meanline_visible=True, fillcolor=fillcolor_i, opacity=0.6,
                               x0=x_label_i), layout=layout)


    fig.update_xaxes(showline=True, linewidth=2, linecolor='#E3E3E3', gridcolor='#E3E3E3', mirror=True )
    fig.update_yaxes(showline=True, linewidth=2, linecolor='#E3E3E3', gridcolor='#E3E3E3', mirror=True)

    
    fig.update_layout(title_text="Violinplot de la variable "+x_label_i, yaxis_zeroline=False)
    fig.show()
```

```{code-cell} ipython3
graph_int_violin(data['saturated_fat_100g'], "#6D8260", "saturated_fat_100g")
```

Il reste des données extrêmes. Mais elles sont cohérentes. Donc nous allons les conserver.

+++

#### Etudions la variable fiber_100g

+++

Pour information, l'aliment le plus riche en fibre est la cannelle avec 43,5g (pour 100g).

+++

source: https://sante.journaldesfemmes.fr/calories/classement/aliments/fibres

```{code-cell} ipython3
data = data.loc[data["fiber_100g"]<=45]
```

Traçons un violinplot pour observer ces modifications

```{code-cell} ipython3
graph_int_violin(data['fiber_100g'], "#6D8260", "fiber_100g")
```

Nous observons que nos données sont concentrées en bas du violinplot.

#### Passons à la variable salt_100g

+++

Traçons un boxenplot qui est comme une boite à moustache sauf qu'il a plus de quantiles.

```{code-cell} ipython3
pk.graph_boxenplot(data, "salt_100g", (0.82, 0.28, 0.09),"Boxenplot du sel contenu (sur 100g) dans les aliments",(8,6))
```

Nous observons que les données sont regroupées à gauche du graphique. Il y a des valeurs très extrêmes.

+++

L'aliment le plus riche en sel est : sel non iodé non fluoré avec 39100 mg

source : https://sante.journaldesfemmes.fr/calories/classement/aliments/sel

+++

Supprimons les données supérieurs à 41g.

```{code-cell} ipython3
data = data.loc[data["salt_100g"]<=40]
```

```{code-cell} ipython3
boxplot_int(data.salt_100g, "salt_100g", "indianred",1 )
```

Il reste des valeurs extrêmes mais nous les conservons car elles sont réelles.

+++

#### Maintenant que nous avons supprimé nos données aberrantes, nous pouvons recalculer nos statistiques et retracer des violinplots

```{code-cell} ipython3
for col in data.select_dtypes(include=['float64']).columns:
    pk.graph_boxplot(data, col, "Boite à moustache de la variable "+col, "#6D8260", (14,8))
```

```{code-cell} ipython3
data.describe().round(2)
```

On a des données à -15 pour les nutritions score, mais cela est possible.
source : https://www.santepubliquefrance.fr/media/files/02-determinants-de-sante/nutrition-et-activite-physique/nutri-score/qr-scientifique-technique-en => voir page 27

Le min est -15 et le max 40.

Nous remarquons que les écart-type de nos variables sont assez faibles et que les boites à moustache sont toutes assez resserrées. Ainsi, nos données sont peu dispersées au sein de nos variables quantitatives.

+++

Nous apprenons plusieurs informations sur nos aliments (sur 100g) :
- 75% des aliments contiennent moins de 2 additifs et non pas d'huile de palme
- 50% des aliments apportent plus de 362 calories (energie en calorie)
- 75 % des aliments contiennent moins de 5g de matières grasses et moins de 1.60g de matières grasses saturées
- 25% des aliments contiennent moins de 4.20g de carbohydrates
- 50% des aliments contiennent moins de 3.50g de sucres et moins de 0.46g de sels
- 25% des aliments ont moins de 0.03g de sodium
- 75% des aliments ont moins de 1.90g de fibres et moins de 6.59g de protéines

De plus, nous savons que 75% de nos aliments ont une note au nutriscore fr inférieure à 5 et que le maximum est 32.

+++

#### Maintenant, étudions les distributions de nos différentes variables.

```{code-cell} ipython3
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
```

```{code-cell} ipython3
for col in data.select_dtypes(include=['float64']).columns:
    hist_int(col, "#AB2300", "Distribution des aliments en fonction de la variable "+col, col, "Fréquence")
```

Nous remarquons que la majorité des distributions des variables sont asymétriques vers la gauche, excepté pour la varible energy_100g qui est plus aplatie.

+++

#### Réalisons un Test de skewness pour confirmer.

Pour rappel : 
- Si y1=0 alors la distribution est symétrique.
- Si y1>0 alors la distribution est étalée à droite.
- Si y1<0 alors la distribution est étalée à gauche.

```{code-cell} ipython3
for col in data.select_dtypes(include=['float64']).columns:
    data_i=data[col].loc[pd.isna(data[col])==False]
    print("Variable : "+col+" ----- y1="+ str(data[col].skew()))
```

Le test de Skewness confirme ce que nous savons vu grâce au histogramme, les distributions sont étalées à droite.

+++

#### Vérifions comment nos variables se comportent par rapport à la loi normale grêce au test de kurtosis.

Si γ2=0 , alors la distribution a le même aplatissement que la distribution normale.

Si γ2>0 , alors elle est moins aplatie que la distribution normale : les observations sont plus concentrées.

Si γ2<0 , alors les observations sont moins concentrées : la distribution est plus aplatie.

```{code-cell} ipython3
for col in data.select_dtypes(include=['float64']).columns:
    data_i=data[col].loc[pd.isna(data[col])==False]
    print("Variable : "+col+" -- y2="+ str(data[col].kurtosis()))
```

Nos données sont toutes plus concentrées que la loi normale.

#### Faisons un test pour vérifier que nos distributions ne suivent pas une loi normale.

+++

Pour rappel, les hypothèse de du test de la loi normal est :
- h0 = la distribution est normale. (p-value > 0.05)
- h1 = la distribution n'est pas normale. (p-value < 0.05)

```{code-cell} ipython3
for col in data.select_dtypes(include=['float64']).columns:
    data_i=data[col].loc[pd.isna(data[col])==False]
    print("Variable: "+col+" ----- "+ str(scipy.stats.normaltest(data_i)))
```

Nous rejetons H0 pour toutes nos variables, la p-value est inférieure à 0.05.
Nos distributions ne suivent pas une loi normale.

+++

### Créons des regroupements afin que nos distributions soient mieux réparties et que nos variables soient plus pertinentes pour la suite.

Nous appliquerons un système de note en fonction des intervalles : 
- si l'apport nutritionnel fait partie des aliments dit positifs : -6, -4,-2,0,2,4,6,8,10 ...Etc
- si l'apport nutritionnel fait partie des aliments dit négatifs : 2,4,6,8,10..Etc

+++

#### Commençons par la variable additives_n

```{code-cell} ipython3
pk.graph_hist(data["additives_n"],[0,2,4,6,8,10,24] ,"Distribution des produits en fonction de la variable additives_n",
              "#6D8260", 0,24, 2, 0, 50000, "additives_n", 'Fréquences',(11,7))
```

```{code-cell} ipython3
def reg_add(x):
    if x<1:
        return 0
    elif x>=1 and x<2:
        return 2
    elif x>=2 and x<4:
        return 4
    elif x>=4 and x<6:
        return 6
    elif x>=6 and x<8:
        return 8
    elif x>=8:
        return 10
```

```{code-cell} ipython3
data['reg_additives']=data.apply(lambda row: reg_add(row.additives_n), axis=1)
```

```{code-cell} ipython3
pk.graph_barplot(data['reg_additives'], "Répartition des aliments en fonction des additifs", 
              (0.82, 0.28, 0.09),
              0, 70, "Intervalle - additives", "Fréquence en %",70, 1,(11,7))
```

Notre distribution est mieux répartie. Conservons ce regroupement.

#### Passons à la variable energy_100g

```{code-cell} ipython3
pk.graph_hist(data["energy_100g"],[0,200,400,600,800,901], "Distribution des produits en fonction de la variable energy_100g","#6D8260",
          0,901, 200, 0, 30000 , "energy_100g", 'Fréquences',(11,7))
```

```{code-cell} ipython3
def reg_energy(x):
    if x<200:
        return 1
    elif x>=200 and x<350:
        return 2
    elif x>=350 and x<500:
        return 3
    elif x>=500 and x<700:
        return 4
    elif x>=700:
        return 5
```

```{code-cell} ipython3
data['reg_energy']=data.apply(lambda row: reg_energy(row.energy_100g), axis=1)
```

```{code-cell} ipython3
pk.graph_barplot(data['reg_energy'], "Répartition des aliments en fonction des calories", 
              (0.82, 0.28, 0.09),
              0, 35, "Classe energie", "Fréquence en %",70, 1,(11,7))
```

Notre distribution est mieux répartie.

#### Etudions la distribution de la variable fat_100g

```{code-cell} ipython3
pk.graph_hist(data["fat_100g"],[0,5,10,15,20,25,100], "Distribution des produits en fonction de la variable fat_100g",
              "#6D8260", 0,100, 5, 0, 65000 , "fat_100g", 'Fréquences',(11,7))
```

```{code-cell} ipython3
def reg_fat(x):
    if x<1:
        return 0
    elif x>=1 and x<4:
        return 2
    elif x>=4 and x<6:
        return 3
    elif x>=6 and x<10:
        return 4
    elif x>=10:
        return 5
```

```{code-cell} ipython3
data['reg_fat']=data.apply(lambda row: reg_fat(row.fat_100g), axis=1)
```

```{code-cell} ipython3
pk.graph_barplot(data['reg_fat'], "Répartition des aliments en fonction des matières grasses", 
              (0.82, 0.28, 0.09),
              0, 50, "Classe fat_100g", "Fréquence en %",70, 1,(11,7))
```

Nous validons ce regroupement.

#### Passons à la variable saturated_fat_100g

```{code-cell} ipython3
pk.graph_hist(data["saturated_fat_100g"],[0,4,8,90], "Distribution des produits en fonction de la variable fat_100g",
             "#6D8260",
          0,100, 5, 0, 85000 , "saturated_fat_100g", 'Fréquences',(11,7))
```

```{code-cell} ipython3
def reg_saturated_fat(x):
    if x<=0:
        return -1
    elif x>=1 and x<4:
        return 4
    elif x>=4 and x<8:
        return 8
    elif x>=8 and x<40:
        return 40
    elif x>=40:
        return 90
```

```{code-cell} ipython3
data['reg_saturated_fat']=data.apply(lambda row: reg_saturated_fat(row.saturated_fat_100g), axis=1)
```

```{code-cell} ipython3
pk.graph_barplot(data['reg_saturated_fat'], "Répartition des aliments en fonction des matières grasses saturées", 
              (0.82, 0.28, 0.09),
              0, 70, "Note saturated_fat", "Fréquence en %",70, 1,(11,7))
```

Nous conservons ce regroupement.

+++

#### Etudions la variable carbohydrates_100g

```{code-cell} ipython3
pk.graph_hist(data["carbohydrates_100g"],[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,100], 
              "Distribution des produits en fonction de la variable carbohydrate_100g",
              "#6D8260",
          0,100, 15, 0, 25000 , "carbohydrates_100g", 'Fréquences',(11,7))
```

```{code-cell} ipython3
def reg_carbohydrates(x):
    if x<5:
        return 1
    elif x>=5 and x<10:
        return 3
    elif x>=15 and  x<25:
        return 5
    elif x>=25:
        return 10
```

```{code-cell} ipython3
data['reg_carbohydrates']=data.apply(lambda row: reg_carbohydrates(row.carbohydrates_100g), axis=1)
```

```{code-cell} ipython3
pk.graph_barplot(data['reg_carbohydrates'], "Répartition des aliments en fonction des carbohydrates", 
              (0.82, 0.28, 0.09),
              0, 40, "Note carbohydrates", "Fréquence en %",70, 1,(11,7))
```

```{code-cell} ipython3
pk.graph_hist(data["fiber_100g"],[0,1,2,3,4,5,6,7,8,100], 
              "Distribution des produits en fonction de la variable fiber_100g",
              "#6D8260",
          0,100, 15, 0, 45000 , "fiber_100g", 'Fréquences',(11,7))
```

```{code-cell} ipython3
def reg_fiber(x):
    if x<1:
        return 2
    elif x>=1 and x<3:
        return -3
    elif x>=3 and x<5:
        return -5
    elif x>=5 and x<7:
        return -10
    elif x>=7:
        return -15
```

```{code-cell} ipython3
data['reg_fiber']=data.apply(lambda row: reg_fiber(row.fiber_100g), axis=1)
```

```{code-cell} ipython3
pk.graph_barplot(data['reg_fiber'], "Répartition des aliments en fonction des fibres", 
              (0.82, 0.28, 0.09),
              0, 60, "Classe fibres", "Fréquence en %",0, 1,(11,7))
```

```{code-cell} ipython3
pk.graph_hist(data["proteins_100g"],[0,10,15,20,50,100], 
              "Distribution des produits en fonction de la variable proteins_100g","#6D8260",
          0,100, 15, 0, 80000 , "proteins_100g", 'Fréquences',(11,7))
```

```{code-cell} ipython3
def reg_protein(x):
    if x<=0:
        return 4
    elif x>=1 and x<2:
        return 1
    elif x>=2 and x<4:
        return 0
    elif x>=4 and x<8:
        return -3
    elif x>=8 and x<15:
        return -4
    elif x>=15:
        return -5
data['reg_protein']=data.apply(lambda row: reg_protein(row.proteins_100g), axis=1)
pk.graph_barplot(data['reg_protein'], "Répartition des aliments en fonction des protéines", 
              (0.82, 0.28, 0.09),
              0, 25, "Note protéine", "Fréquence en %",0, 1,(11,7))
```

Nous utiliserons ce regroupement.

+++

## Conclusion des variables quantitatives

+++

Nous avons vu que nos données étaient peu dispersées au sein des variables quantitatives.
Et que les distributions étaient de forme asymétrique vers la gauche.

Nous avons appris que nos aliments étaient assez bien notés au nutriscore fr. En effet, 75% des aliments ont une note inférieure à 5 sachant que le maximum est 41. 

Nous avons aussi pu décrire plus précisément les nutriments de nos aliments : 
- 50% des aliments apportent plus de 362 calories (energie en calorie)
- 75 % des aliments contiennent moins de 5g de matières grasses et moins de 1.60g de matières grasses saturées
- 25% des aliments contiennent moins de 4.20g de carbohydrates
- 50% des aliments contiennent moins de 3.50g de sucres et moins de 0.46g de sels
- 75% des aliments ont moins de 1.90g de fibres et moins de 6.59g de protéines


Nous avons dans notre base de données des aliments peu calorique, peu gras et peu salé. Mais ils contiennent peu de fibre et peu de protéines.

+++

Regardons à présent ce que peut nous apprendre les variables qualitatives sur notre population.

+++

### Variables qualitatives

```{code-cell} ipython3
data.info()
```

```{code-cell} ipython3
data[data.select_dtypes(include=['object', 'string']).columns].describe(include='all').round(2)
```

Nous remarquons que la plupart de nos variables qualitatives comportent beaucoup de catégories et que le mode est "N.A" (non renseigné).
Nous observons aussi que le mode est la modalité B pour le nutriscore Fr et la modalité "NON" pour la variable "ingredients_from_palm_oil".

Etudions chaque variable pour obtenir plus d'information sur leur répartition.

+++



+++

#### Commençons par la variable product_name

+++

Cette variable contient quasiment autant d'individus que de catégories. 

Regardons le top 10.

```{code-cell} ipython3
data["product_name"].value_counts(normalize=True).head(10)
```

Comme nous l'avons remarqué précédemment, nos données sont très dispersées au sein de cette variable.

+++

#### Regardons la variable quantity

+++

Cette variable contient 3906 catégories. 

Pour rappel, ces données nous renseigne sur la quantité contenu dans les produits (avec l'unité utilisé)

```{code-cell} ipython3
data["quantity"].value_counts(normalize=True).head(10)
```

La modalité "AUTRES" représente 75% des aliments.

Retirons cette modalité et regardons à nouveau le top 10.

```{code-cell} ipython3
data_autres=data.loc[data['quantity']!="AUTRES"]
```

```{code-cell} ipython3
data_autres["quantity"].value_counts(normalize=True).head(10)
```

3.7% des aliments contiennent 300g et 1% des aliments contiennent 600g.

Les données sont très éparpillées au sein des modalités et sont à des échelles différentes. Elle nous apporte pas d'information supplémentaire, nous pouvons la supprimer.


```{code-cell} ipython3
del data["quantity"]
```

#### Etudions les variables packaging et packaging_tags

+++

La variable packaging contient 4305 catégories et packaging_tag en contient 3592.

+++

Ces variables contiennent aussi beaucoup de catégorie. Etudions les top 10.

```{code-cell} ipython3
data["packaging"].value_counts(normalize=True).head(10)
```

Les données sont très éparpillées au sein des catégories. Et la variable Autres est surreprésentée pour les 2 catégories (78%)
Nous pouvons supprimer ces variables

+++

Excluons la catégorie "AUTRES"

```{code-cell} ipython3
data_autres=data.loc[data['packaging']!="AUTRES"]
data_autres["packaging"].value_counts(normalize=True).head(10)
```

```{code-cell} ipython3
data_autres=data.loc[data['packaging_tags']!="AUTRES"]
data_autres["packaging_tags"].value_counts(normalize=True).head(10)
```

Les catégories ne sont pas très pertinentes pour ces deux variables. En effet, nous avons 3% des aliments qui sont dans des conserves et 0.09% des aliments qui sont dans des conserves,métal. Nous pouvons les exclure.

```{code-cell} ipython3
del data["packaging"]
del data["packaging_tags"]
```

```{code-cell} ipython3
data["brands"].nunique()
```

```{code-cell} ipython3

max(data_autres["brands"].value_counts(normalize=True))
```

Créons une fonction qui supprime les variables qui ont un seuil de catégorie important et qui sont très dispersées.

```{code-cell} ipython3
def del_quali_norelevant(data,identifiant,  delete_i, del_seuilcat_i, del_seuilfreq_i, all_i):
    info=""
    for col in data.select_dtypes(include=['object']).columns:
        if type(data[col].loc[pd.isna(data[col])==False].iloc[0])== str and col not in identifiant:
                data_autres=data.loc[data[col]!="AUTRES"]
                if delete_i==1 and max(data_autres[col].value_counts(normalize=True).head(10))<del_seuilfreq_i and (data[col].nunique()>del_seuilcat_i):
                    print("deleted "+col)
                    del data[col]
                print("Variable :"+ col + "----Max freq :"+str(max(data_autres[col].value_counts(normalize=True))))      

    
```

On supprime les colonnes avec plus de 500 catégorie et dont le maximum de la fréquence d'une modalité est inférieur à 0.05

```{code-cell} ipython3
del_quali_norelevant(data,["code", "created_datetime", "product_name", "countries", "countries_tags",
                          "ingredients_text"], 1, 500, 0.05, 0)
```

```{code-cell} ipython3
data.columns
```

```{code-cell} ipython3
data[data.select_dtypes(include=['object', 'string']).columns].describe(include='all').round(2)
```

#### Etudions la variable main_category

```{code-cell} ipython3
data_autres=data.loc[data['main_category']!="AUTRES"]
```

```{code-cell} ipython3
data_autres["main_category"].value_counts(normalize=True).head(10)
```

```{code-cell} ipython3
cat_top= data["main_category"].value_counts(normalize=True).head(10).reset_index(name="values")
```

```{code-cell} ipython3
cat_top["index"]
```

```{code-cell} ipython3
data_cat = data_autres.loc[ data['main_category'].isin(cat_top["index"])==True]
```

```{code-cell} ipython3
data[data.select_dtypes(include=['object', 'string']).columns].describe(include='all').round(2)
```

```{code-cell} ipython3
pk.graph_bubbleplot(data_cat['main_category'], 'g',"main_category", "Fréquence en %", 
                 "Répartition des aliments par catégorie (top 10)", 0, 23,
                    "black", "center", "bold", (14,8),90)
```

La partie Autres est surreprésentée (77%).

Si on exclut la modalité Autres, les données sont très dispersées au sein de cette variable.
Nous avons 8% des aliments qui appartiennent à la catégorie canned-foods  (aliment en conserve) et 2% des aliments sont des boissons et 1.5% des aliments surgelés.

+++

#### Regardons la variable ingredients_text

```{code-cell} ipython3
data["ingredients_text"].head(5)
```

Cette variable contient la liste des ingredients contenus dans les aliments.
Il n'est donc pas pertinent de l'étudier

+++

#### Etudions les variables states, states_tags et state_fr

```{code-cell} ipython3
data["states"].unique()
```

Ce sont des informations sur l'état de la fiche aliment. Nous pouvons supprimer ces variables.

```{code-cell} ipython3
data.drop(["states", "states_tags", "states_fr"], axis=1, inplace=True)
```

#### Observons les variables additives_tags et	additives_fr

+++

Cette variable comporte 11 748 catégories

+++

Regardons le top 10.

```{code-cell} ipython3
data_autres["additives_tags"].value_counts(normalize=True).head(10)
```

```{code-cell} ipython3
data_autres["additives_fr"].value_counts(normalize=True).head(10)
```

La variable "additives_fr" est la "traduction" du code des additifs. Elles ont donc une répartition identiques.

+++

Les données sont très dispersées au sein de ces variables. La catégorie Autres représentent 50% des aliments.

De plus, nous avons une colonne qui contient le nombre d'additifs, et ces colonnes ne sont pas très parlantes.
Nous décidons de les supprimer.

```{code-cell} ipython3
data.drop(["additives_fr", "additives_tags"], axis=1, inplace=True)
```

#### Analysons à présent la variable nutrition_grade_fr

```{code-cell} ipython3
pk.graph_barplot(data['nutrition_grade_fr'], "Répartition des produits selon le nutrition grade", 
              (0.82, 0.28, 0.09),
              0, 40, "nutrition_grade_fr", "Fréquence en %",0,1, (11,7))
```

Nous observons que nous avons très peu d'aliments de catégorie E (moins de 5%) et de catégorie D (environs 10%)

+++

#### Etudions la variable ingredients_from_palm_oil

+++

Traçons un graphique pour observer la répartition des aliments au sein de cette variable

```{code-cell} ipython3

t = pd.crosstab(data.ingredients_from_palm_oil, "freq", normalize=True)
t = t.assign(column = t.index, freq = 100 * t.freq)
```

```{code-cell} ipython3
fig = px.pie(t, t.column, t.freq , color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()
```

Nous remarquons que seulement 0.7% des aliments contiennent potentiellement de l'huile de palme.

+++

#### Passons aux variables pnns_groups_1 et pnns_groups_2

+++

Traçons un graphique circulaire pour étudier la répartition des aliments au sein de ces variables.

```{code-cell} ipython3
import plotly.express as px
import numpy as np
fig = px.sunburst(data, path=['pnns_groups_1', 'pnns_groups_2'])
fig.show()
```

Excluons la catégorie "Autres" qui représente 57% des aliments et renommons la catégorie "Unknown" en "AUTRES".

```{code-cell} ipython3
data['pnns_groups_1'].unique()
```

```{code-cell} ipython3
data.head(20)
```

```{code-cell} ipython3
data['pnns_groups_1'] = data['pnns_groups_1'].str.replace("UNKNOWN",'AUTRES')
data['pnns_groups_2'] = data['pnns_groups_2'].str.replace("UNKNOWN",'AUTRES')
data_autres=data.loc[(data.pnns_groups_1!="AUTRES") & (data.pnns_groups_2!="AUTRES") ] 
```

```{code-cell} ipython3
import plotly.express as px
import numpy as np
fig = px.sunburst(data_autres, path=['pnns_groups_1', 'pnns_groups_2'])
fig.show()
```

Nous remarquons que la plupart des aliments sont des boissons et des ingredients.

+++

#### Décrivons nos données en regardons le nombre d'aliments par date de création des fiches.

```{code-cell} ipython3

data['year'] = pd.DatetimeIndex(data['created_datetime']).year
data['month'] =pd.DatetimeIndex(data['created_datetime']).month
data["year_month"]=pd.DatetimeIndex(data['created_datetime']).strftime("%Y/%m")
```

```{code-cell} ipython3
data
```

```{code-cell} ipython3
data.columns
```

```{code-cell} ipython3
data_month=data[["year", "month", "nutrition_grade_fr"]].value_counts().reset_index(name="value")
```

```{code-cell} ipython3
data_month
```

```{code-cell} ipython3


fig = px.scatter(data_month, x="month", y="value", animation_frame="year", animation_group="nutrition_grade_fr",
           color="nutrition_grade_fr", 
           log_x=True, size_max=55)

fig["layout"].pop("updatemenus") # optional, drop animation buttons
fig.show()
```

Nous remarquons que la plupart de nos données ont été créées en mars 2017.

+++

#### Regardons d'où provient la majorité des aliments grâce à la variable countries.

+++

Nous avons besoin d'importer les pays et leur code iso.

```{code-cell} ipython3
info_countries=pd.read_json(".\data\countries.json")
```

```{code-cell} ipython3
info_countries
```

```{code-cell} ipython3
info_countries=pk.data_majuscule(info_countries)
```

```{code-cell} ipython3
info_countries = info_countries.rename(columns={'name': 'countries_tags'})
```

```{code-cell} ipython3
data.countries_tags.unique()
```

```{code-cell} ipython3
data.countries_tags=data.countries_tags.str.replace("EN:","")
```

```{code-cell} ipython3
data.countries_tags=data.countries_tags.str.replace("UNITED-STATES","UNITED STATES OF AMERICA")
```

```{code-cell} ipython3
data.countries_tags
```

!nous pouvons à présent ajouter lke code iso dans notre base de données.

```{code-cell} ipython3
data_merge = pd.merge(data, info_countries, how="left", on=["countries_tags"], indicator=True,  suffixes=('', '_del'))
```

```{code-cell} ipython3
data_merge
```

```{code-cell} ipython3
data_merge.alpha3
```

```{code-cell} ipython3
data_result = data_merge.loc[data_merge["_merge"] == "both"].drop("_merge", axis=1)
data_result = data_result[[c for c in data_result.columns if not c.startswith('id') |c.startswith('alpha2') | c.endswith('_del') ]]
```

```{code-cell} ipython3
data_result.alpha3.unique()
```

```{code-cell} ipython3
data= data_result
```

```{code-cell} ipython3
data.alpha3.unique()
```

```{code-cell} ipython3
maptest=data[["alpha3", "nutrition_grade_fr", "countries_tags"]].value_counts().reset_index(name="value")
```

```{code-cell} ipython3
maptest
```

```{code-cell} ipython3
import plotly.express as px

fig = px.scatter_geo(maptest, locations="alpha3", color="nutrition_grade_fr",
                     hover_name="countries_tags", size="value",
                     projection="natural earth")
fig.show()
```

La majorité de nos aliments proviennent des USA.

+++

#### Analysons à présent la variable nutrition_grade_fr

```{code-cell} ipython3
pk.graph_barplot(data['nutrition_grade_fr'], "Répartition des produits selon le nutrition grade", 
              (0.82, 0.28, 0.09),
              0, 40, "nutrition_grade_fr", "Fréquence en %",0,1, (11,7))
```

Nous observons que nous avons très peu d'aliments de catégorie E (moins de 5%) et de catégorie D (environs 10%)

```{code-cell} ipython3

```

## Conclusion qualitatives

+++

Nous avons appris que les variables qualitatives étaient très disperséesau sein des catégories et que la majorité des aliments étaient dans la catégorie "AUTRES".

Nous avons aussi remarqué que nos données ont été créées en mars 2017 et que les aliments proviennent des USA.
Nous savons aussi que nos aliments sont bien notés, 32% des aliments ont un B et 28% des aliments ont un A contre moins de 5% des aliments ont un E.

+++

Maintenant que nous connaissons nos données. Il est important de pouvoir faire un zoom sur le nutriscore afin d'améliorer la santé des français.

+++

# Est-ce que le nutriscore contribue à améliorer la santé des français ? Dans ce but, est ce que le nutriscore permet de manger équilibrer ?

+++

## Nos variables

+++


Nous allons commencer par sélectionner les colonnes qu'il nous faut pour améliorer la santé des français.

Certains organismes officiels, comme l’AFSSA (Agence française de sécurité sanitaire des aliments), ont crée des recommandations sous forme d’apports nutritionnels conseillés (ANC) pour chaque type de nutriment.

source : https://www.vidal.fr/sante/nutrition/equilibre-alimentaire-adulte/recommandations-nutritionnelles-adulte/en-pratique.html

Selon le tableau nutritionnel, voici les informations dont nous avons besoin : Glucides, Lipides, Protéines, Fibres. Et les aliments qui contiennent de l'huile de palme sont à notifier.

Comment le Nutri-Score Fr est-il calculé?

Points négatifs : l'énergie, les graisses saturées, les sucres, et le sel (des niveaux élevés sont considérés comme mauvais pour la santé)

Points positifs : la proportion de fruits, de légumes, de noix, d'huiles d'olive, de colza et de noix, de fibres et de protéines (les niveaux élevés sont considérés comme bons pour la santé).

Sélectionnons donc seulement les colonnes nécessaires à notre étude. Nous sélectionnons tous les aliments même ceux qui ne sont pas vendu en France. En effet, ils peuvent être vendus prochainement et être interessant pour notre étude.

```{code-cell} ipython3
data.columns
```

```{code-cell} ipython3
data_study = data[["code", "product_name",  
            "ingredients_text","energy_100g", "carbohydrates_100g", "proteins_100g", "fat_100g",
            "saturated_fat_100g", "fiber_100g", "sugars_100g", "salt_100g",
             "additives_n", "ingredients_from_palm_oil",
            "ingredients_from_palm_oil_n", "ingredients_that_may_be_from_palm_oil_n", 
            "nutrition_grade_fr", "main_category", "nutrition_score_fr_100g", 'reg_additives',
       'reg_energy', 'reg_fat', 'reg_carbohydrates', 'reg_fiber', 
       'reg_protein']]
```

## Quels sont ses caractéristiques ?

#### Techniquement, voici les éléments qui agit sur le nutriscore :

Points négatifs : l'énergie, les graisses saturées, les sucres, et le sel (des niveaux élevés sont considérés comme mauvais pour la santé)

Points positifs : la proportion de fruits, de légumes, de noix, d'huiles d'olive, de colza et de noix, de fibres et de protéines (les niveaux élevés sont considérés comme bons pour la santé).

+++

## Est - ce que si nous mangeons que des aliments avec un nutriscore A, serons - nous en bonne santé ?

+++

### Comparons les informations  des différentes notes.

+++

Excluons les données non renseignées du Nutriscore.

```{code-cell} ipython3
dataN = data_study.loc[data["nutrition_grade_fr"]!="AUTRES"].sort_values(by=["nutrition_grade_fr"])
```

### Analyse bivariée : Etudions le lien entre la variable nutrition_grade_fr et les autres variables.

```{code-cell} ipython3
dataN.describe()
```

```{code-cell} ipython3
def percentile(n):
    def percentile_(x):
        y=x.loc[pd.isna(x)==False]
        return np.percentile(y, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_
```

```{code-cell} ipython3
for col in data_study.select_dtypes(include=['float64']).columns:
    print(col)
    print(dataN.groupby("nutrition_grade_fr")[col].agg([np.median,np.min, np.max, percentile(25), percentile(75)]))
```

```{code-cell} ipython3
for col in data_study.select_dtypes(include=['float64']).columns:
    print("moyenne "+col +" = "+str(dataN[col].mean()))
    pk.graph_boxplot_by_group(dataN, col,"nutrition_grade_fr", "Boite à moustache de la variable "+col+" en fonction du nutriscore", "rocket_r", (14,8))
```

Grâce à ces boites à moustache, nous remarquons qu'ils restent des données extrêmes. Mais dans ce contexte, nous devons les conserver pour ne pas perdre d'information.

Les boites à moustache sont peu dispersées même si en fonction des notes, leur dispersion peuvent plus ou moins varié.

+++

Comment pouvons - nous décrire les aliments avec une note A ?
- riches en fibres et en proteines
- peu de sel, de sucre et de matières grasses (y compris en graisses saturés)
- contiennent des fruits, végétaux ou des arachides
- faible en additifs
- apport en energie proche de la moyenne

Comment pouvons - nous décrire les aliments avec une note E ?
- riches en sucre, en additif et en glucides
- pauvres en fibres, proteines, sel
- une energie qui varie beaucoup dans les extrêmes.

```{code-cell} ipython3
dataN.columns
```

```{code-cell} ipython3
sns.catplot(x = "nutrition_grade_fr", hue = "ingredients_from_palm_oil", data = dataN, kind = "count", color=(0.82, 0.28, 0.09))
```

Les produits qui contiennent potentiellement de l'huile de palme ont les meilleurs notes.

+++

### Confirmons nos résultats précédents en vérifiant le lien entre nos variables quantitatives et le grade du nutriscore

+++

Pour cela, nous pourrions modéliser nos données en réalisant une ANOVA.
Mais cela suppose 3 choses : 
- l'indépendance entre chaque groupe
- l'égalité des variances
- la normalité des résidus (cela permet de ne pas affirmer qu'il existe une différence de moyenne entre les groupes qui serait causée par le hasard).

+++

### L'indépendance entre nos groupes

+++

Selon le contexte, nous savons que chaque lettre du nutriscore est indépendante car elle représente un classement des aliments.

+++

### L'égalité des variances

+++

Réalisons un test de bartlette afin de confirmer ce que nous avons vu lors de l'analyse bivariée.

H0 : Les variances de chaque groupe sont égales si p-value > 5%

H1 : Les variances de chaque groupe ne sont pas toutes égales < 5%

+++

Définissons nos groupes

```{code-cell} ipython3
data_a=dataN.loc[data["nutrition_grade_fr"]=="A"]
data_b=dataN.loc[data["nutrition_grade_fr"]=="B"]
data_c=dataN.loc[data["nutrition_grade_fr"]=="C"]
data_d=dataN.loc[data["nutrition_grade_fr"]=="D"]
data_e=dataN.loc[data["nutrition_grade_fr"]=="E"]
```

Effectuons le test pour chaque variable quantitative

```{code-cell} ipython3
for col in dataN.select_dtypes(include=['float64']).columns:
    print("colonne "+col+" "+str(scipy.stats.bartlett(data_a[col], data_b[col], data_c[col], data_d[col], data_e[col])))
```

Nous rejetons h0 pour toutes nos variables. Les variances ne sont pas égales.

+++

La deuxième condition pour effectuer une ANOVA n'est pas validée.

```{code-cell} ipython3
import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols('energy_100g ~ nutrition_grade_fr ', data=dataN).fit()
anova = sm.stats.anova_lm(model, typ=2)

anova
```

H0 : les moyennes sont équivalent dans les groupes
H1 : les moyennes sont différentes
    
P<0.05 on rejette H0. et on admet H1.

#### Les résidus doivent suivre une loi normale

```{code-cell} ipython3
import numpy as np
import statsmodels.api as sm
import pylab
from scipy.stats import shapiro
model = ols('energy_100g ~ nutrition_grade_fr', data=dataN).fit()
scipy.stats.normaltest(model.resid)
```

~H0 : Les résidus suivent une loi normale si p-value > 5%~
H1 : Les résidus ne suivent pas une loi normale si p-value < 5%

```{code-cell} ipython3
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pylab
from scipy.stats import shapiro
model = ols('energy_100g ~ nutrition_grade_fr', data=dataN).fit()


sm.qqplot(model.resid, line='45')
pylab.show()
```

### Utilisons le test de Krusdall (utilise les rangs au lieu des moyennes)

+++

H0 :  la médiane de la population de tous les groupes est égale
H1 :  la médiane de la population d'au moins un groupe n'est pas égale

+++

Test Krusdall(utilise mediane et quantile)

```{code-cell} ipython3
for col in data.select_dtypes(include=['float64']).columns:
    print("colonne "+col+": "+ str(scipy.stats.kruskal(*[group[col] for name, group in dataN.groupby("nutrition_grade_fr")])))
```

Signatifivement différent on rejette H0 car pvalue<5%

+++

Selon le test, la mediane d'un groupe sont significativement différents pour chacunes des variables.

+++

## Conclusion Nutriscore

+++

 les aliments avec une note A ?
- riches en fibres et en proteines 
- peu de sel, de sucre et de matières grasses (y compris en graisses saturés)
- contiennent des fruits, végétaux ou des arachides
- faible en additifs
- apport en energie proche de la moyenne

 les aliments avec une note E ?
- riches en sucre, en additif et en glucides
- pauvres en fibres, proteines, sel
- une energie qui varie beaucoup dans les extrêmes.


Cependant, nous avons besoin d'une alimentation équilibrée. 

Voici l'apport quotidien pour un adulte : 

Glucides = 250g (carbohydrates)

lipides = 50 insaturé et 20 sature

Protéines =	45 g pour une personne de 55 kg
            60 g pour une personne de 75 kg
            
Fibres =	25 à 30 g dont une moitié issue des céréales et l'autre issue des fruits et légumes
    
Donc si nous mangeons que des produits A, cela ne nous permet pas de manger équilibrer et ne permet pas d'améliorer la santé.
Comment pouvons - nous améliorer la santé des français?

+++

# Objectif une meilleure santé pour tous : Application Smart Food

+++

Un classement des aliments et le calcul de leur combinaison permettront d'améliorer la santé des français et les guider vers une alimentation équilibrée sans se priver.

+++

## Classement des aliments

+++

Etudions de plus près nos aliments et regardons qu'elles sont les variables qui permettent d'expliquer au mieux cette population.


Pour commencer, il nous faut exclure les données manquantes et étudier les variables quantitatives qui sont corrélées.

```{code-cell} ipython3
data.columns
```

Regardons les variables corrélées.

```{code-cell} ipython3
sns.set(rc={'figure.figsize':(10,4)})

data_corr = data.corr()

ax = sns.heatmap(data_corr, xticklabels = data_corr.columns , 
                 yticklabels = data_corr.columns, cmap = 'coolwarm')
```

```{code-cell} ipython3
data_acp=data.loc[(pd.isna(data.energy_100g)==False) & (pd.isna(data.carbohydrates_100g)==False) & (pd.isna(data.proteins_100g)==False) & (pd.isna(data.fat_100g)==False)
       & (pd.isna(data["fiber_100g"])==False)
& (pd.isna(data["nutrition_score_fr_100g"])==False)
                 & (pd.isna(data["nutrition_grade_fr"])==False)
                 & (pd.isna(data["additives_n"])==False)]
```

Nous excluons la variable sugars_100 car elle est corrélée avec carbohydrates_100g (glucides).
Et nous excluons la variable satured_fat qui est corrélée avec fat_100g.
Nous excluons aussi la variable energy_100g qui est corrélé avec fat_100g.

```{code-cell} ipython3
data.columns
```

```{code-cell} ipython3
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale
# suppression des colonnes non numériques
WGI_num0 = data_acp.drop(columns =data.select_dtypes(include=['object', 'string']).columns)
```

```{code-cell} ipython3
WGI_num0.columns
```

```{code-cell} ipython3
WGI_num0 = WGI_num0.drop(columns =["reg_additives", "reg_energy", "reg_fat", "reg_saturated_fat", "reg_carbohydrates", "reg_fiber",
                                  "reg_protein", "year", "month", "nutrition_score_fr_100g","nutrition-score-uk_100g",
                                  "energy_100g", "saturated_fat_100g", "sodium_100g", "sugars_100g"])
```

```{code-cell} ipython3
WGI_num0.describe(include="all")
```

```{code-cell} ipython3
pca = PCA()
WGI_num = pca.fit_transform(scale(WGI_num0))

pca.fit(WGI_num)
```

```{code-cell} ipython3
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
```

```{code-cell} ipython3
eig = pd.DataFrame(
    {
        "Dimension" : ["Dim" + str(x + 1) for x in range(8)], 
        "Variance expliquée" : pca.explained_variance_,
        "% variance expliquée" : np.round(pca.explained_variance_ratio_ * 100),
        "% cum. var. expliquée" : np.round(np.cumsum(pca.explained_variance_ratio_) * 100)
    }
)
eig
```

```{code-cell} ipython3
eig.plot.bar(x = "Dimension", y = "% variance expliquée") # permet un diagramme en barres
plt.text(5, 18, "14%") # ajout de texte
plt.axhline(y = 14, linewidth = .5, color = "dimgray", linestyle = "--") # ligne 17 = 100 / 6 (nb dimensions)
plt.show()
```

```{code-cell} ipython3
WGI_pca = pca.transform(WGI_num)
```

```{code-cell} ipython3
# Transformation en DataFrame pandas
WGI_pca_df = pd.DataFrame({
    "Dim1" : WGI_pca[:,0], 
    "Dim2" : WGI_pca[:,1],
    "Dim3" : WGI_pca[:,2],
    "Dim4" : WGI_pca[:,3],
    "Dim5" : WGI_pca[:,4],
    "Dim6" : WGI_pca[:,5],
    "product": data_acp["product_name"],
    "nutrition_grade_fr" : data_acp["nutrition_grade_fr"]
})

# Résultat (premières lignes)
WGI_pca_df.head()
```

```{code-cell} ipython3
WGI_pca_df.plot.scatter("Dim1", "Dim2") # nuage de points
plt.xlabel("Dimension 1 (35%)") # modification du nom de l'axe X
plt.ylabel("Dimension 2 (30%)") # idem pour axe Y
plt.suptitle("Premier plan factoriel (65%)") # titre général
plt.show()
```

```{code-cell} ipython3
WGI_num.shape
```

```{code-cell} ipython3
WGI_num
```

```{code-cell} ipython3
n = WGI_num.shape[0] # nb individus
p = WGI_num.shape[1] # nb variables
eigval = (n-1) / n * pca.explained_variance_ # valeurs propres
sqrt_eigval = np.sqrt(eigval) # racine carrée des valeurs propres
corvar = np.zeros((p,p)) # matrice vide pour avoir les coordonnées
for k in range(p):
    corvar[:,k] = pca.components_[k,:] * sqrt_eigval[k]
# on modifie pour avoir un dataframe
coordvar = pd.DataFrame({'id': WGI_num0.columns, 'COR_1': corvar[:,0], 'COR_2': corvar[:,1], 'COR_3': corvar[:,2]
                        , 'COR_4': corvar[:,3], 'COR_5': corvar[:,4], 'COR_6': corvar[:,5]})
coordvar
```

```{code-cell} ipython3
from yellowbrick.datasets import load_concrete
from yellowbrick.features import PCA
from yellowbrick.style import set_palette

# Load the concrete dataset
visualizer = PCA(scale=True, proj_features=True)
visualizer.fit_transform(WGI_num0)
visualizer.show()
```

```{code-cell} ipython3
ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=WGI_pca_df["Dim1"], 
    ys=WGI_pca_df["Dim2"], 
    zs=WGI_pca_df["Dim3"],
    cmap='tab10'
)
ax.set_xlabel('Dim 1 (35%)')
ax.set_ylabel('Dim 2 (30%)')
ax.set_zlabel('Dim 3 (21%)')
plt.show()
```

```{code-cell} ipython3
WGI_num0
```

```{code-cell} ipython3
test_a=data_acp.loc[data_acp["nutrition_grade_fr"]=="A"].sample(1000)
test_b=data_acp.loc[data_acp["nutrition_grade_fr"]=="B"].sample(1000)
test_c=data_acp.loc[data_acp["nutrition_grade_fr"]=="C"].sample(1000)
test_d=data_acp.loc[data_acp["nutrition_grade_fr"]=="D"].sample(1000)
test_e=data_acp.loc[data_acp["nutrition_grade_fr"]=="E"].sample(1000)
```

```{code-cell} ipython3
frames = [test_a, test_b, test_c, test_d, test_e]
```

```{code-cell} ipython3
test = pd.concat(frames)
```

```{code-cell} ipython3
test.shape
```

```{code-cell} ipython3
test2=test.drop(columns =data_acp.select_dtypes(include=['object', 'string']).columns)
```

```{code-cell} ipython3
test2 = test2.drop(columns =["reg_additives", "reg_energy", "reg_fat", "reg_saturated_fat", "reg_carbohydrates", "reg_fiber",
                                  "reg_protein", "year", "month", "nutrition_score_fr_100g","nutrition-score-uk_100g",
                                  "energy_100g", "saturated_fat_100g", "sodium_100g", "sugars_100g"])
```

```{code-cell} ipython3
test2.describe(include="all")
```

```{code-cell} ipython3
#librairies pour la CAH
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
#générer la matrice des liens
Z = linkage(test2,method='ward',metric='euclidean')
#affichage du dendrogramme
plt.title("CAH")
dendrogram(Z,labels=test2.index,color_threshold=0)
plt.show()
```

Noous décidons de couper  groupes.

```{code-cell} ipython3
from sklearn.cluster import KMeans

kmeans2 = KMeans(n_clusters = 5)
kmeans2.fit(scale(WGI_num0))
```

```{code-cell} ipython3
kmeans2.labels_
```

```{code-cell} ipython3
pd.Series(kmeans2.labels_).value_counts()
```

```{code-cell} ipython3
kmeans2.cluster_centers_
```

```{code-cell} ipython3
WGI_k2 = WGI_num0.assign(classe = kmeans2.labels_)
WGI_k2.groupby("classe").mean()
```

```{code-cell} ipython3
WGI_k2
```

```{code-cell} ipython3
WGI_pca_k2 = WGI_pca_df.assign(classe = kmeans2.labels_)
WGI_pca_k2.plot.scatter(x = "Dim1", y = "Dim2", c = "classe", cmap = "Accent")
plt.show()
```

```{code-cell} ipython3
WGI_pca_k2.shape
```

```{code-cell} ipython3
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
```

```{code-cell} ipython3
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters = k, init = "random", n_init = 20).fit(WGI_num0)
    inertia = inertia + [kmeans.inertia_]
inertia = pd.DataFrame({"k": range(1, 11), "inertia": inertia})
inertia.plot.line(x = "k", y = "inertia")
plt.scatter(2, inertia.query('k == 2')["inertia"], c = "red")
plt.scatter(4, inertia.query('k == 4')["inertia"], c = "red")
plt.show()
```

coude plus marqué au niveau du 4

```{code-cell} ipython3
WGI_pca_k2
```

Ces 4 classes vont permettre de manger équilibrer en les combinant.

+++

Comparons les nouveaux groupes et le nutriscore.

```{code-cell} ipython3
pd.crosstab(WGI_pca_k2.classe, WGI_pca_k2.nutrition_grade_fr, normalize = True)
```

```{code-cell} ipython3
sns.heatmap(pd.crosstab(WGI_pca_k2.nutrition_grade_fr, WGI_pca_k2.classe, normalize = True))
```

```{code-cell} ipython3
t = pd.crosstab(WGI_pca_k2.nutrition_grade_fr, WGI_pca_k2.classe, normalize = "columns")
t = t.assign(nutrition_grade_fr = t.index)
tm = pd.melt(t, id_vars = "nutrition_grade_fr")
tm = tm.assign(value = 100 * tm.value)

sns.catplot("nutrition_grade_fr", y = "value", col = "classe", data = tm, kind = "bar")
```

Aucun groupe ne semble se demarquer, excepté la classe 2.

Les autres groupes sont homogènes. Ce découpage ne semble pas adapter et est assez aléatoire.

+++

## Utilisation des variables quantitatives contenant les notes

+++

Essayons de refaire les calculs avec nos variables "regroupement"

+++

Nous pouvons créer un score total qui ajoutera un poids en fonction des nutriments positifs et négatifs établis par l'agence du nutriscore.

Pour rappel :
- Points négatifs : l'énergie, les graisses saturées, les sucres, et le sel (des niveaux élevés sont considérés comme mauvais pour la santé)

- Points positifs : la proportion de fruits, de légumes, de noix, d'huiles d'olive, de colza et de noix, de fibres et de protéines (les niveaux élevés sont considérés comme bons pour la santé).

```{code-cell} ipython3
data["score_total"] = data.reg_saturated_fat + data['sugars_100g']*data['salt_100g'] + data['reg_additives']*2 + data['reg_energy'] * data['reg_fat'] + data['reg_carbohydrates'] + data['reg_fiber']*4 + data['reg_protein']*4
```

```{code-cell} ipython3
pk.graph_hist(data["score_total"],[-28,-10,-5,0,2,4,6,8,10,15,20,24,50,80,100,150,200,250] ,"Distribution des produits en fonction de la variable score_total",(0.82, 0.28, 0.09),
          -28,2500, 2, 0, 10000, "score_total", 'Fréquences',(11,7))
```

Réalisons un regroupement et appliquons des notes.

```{code-cell} ipython3
def reg_score_total(x):
    if x<-10:
        return -10
    elif x>=-10 and x<-5:
        return -5
    elif x>=-5 and x<1:
        return 0
    elif x>=0 and x<5:
        return 5
    elif x>=5 and x<10:
        return 10
    elif x>=10 and x<15:
        return 15
    elif x>=15 and x<25:
        return 20
    elif x>=25 and x<50:
        return 50
    elif x>=50 and x<80:
        return 80
    elif x>=80 and x<200:
        return 200
    elif x>=200:
        return 300
data['reg_score_total']=data.apply(lambda row: reg_score_total(row.score_total), axis=1)
```

```{code-cell} ipython3
pk.graph_barplot(data['reg_score_total'], "Répartition des aliments en fonction du score total", 
              (0.82, 0.28, 0.09),
              0, 25, "Classe proteine", "Fréquence en %",0, 1,(11,7))
```

```{code-cell} ipython3
data_study.columns
```

```{code-cell} ipython3
data_study = data[["code", "product_name",  "score_total",
            "ingredients_text","energy_100g", "carbohydrates_100g", "proteins_100g", "fat_100g",
            "saturated_fat_100g", "fiber_100g", "sugars_100g", "salt_100g",
             "additives_n", 
            "ingredients_from_palm_oil_n", "ingredients_that_may_be_from_palm_oil_n", 
            "nutrition_grade_fr", "main_category", "nutrition_score_fr_100g", 'reg_additives',
       'reg_energy', 'reg_fat', 'reg_carbohydrates', 'reg_fiber', "reg_score_total",
       'reg_protein']]
data_study=data_study.loc[(pd.isna(data_study.reg_additives)==False) & (pd.isna(data_study.reg_energy)==False) 
               & (pd.isna(data_study.reg_carbohydrates)==False) & (pd.isna(data_study.reg_fat)==False) 
               & (pd.isna(data_study.reg_fiber)==False)  & (pd.isna(data_study.score_total)==False)
                            & (pd.isna(data_study.nutrition_score_fr_100g)==False)
                          & (pd.isna(data_study.reg_protein)==False) ]
```

```{code-cell} ipython3

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale
# suppression des colonnes non numériques
WGI_num0 = data_study.drop(columns = [ "main_category", "nutrition_grade_fr","energy_100g","reg_energy",
                                      "carbohydrates_100g", "proteins_100g", "fat_100g", "fiber_100g",
                                      "additives_n","nutrition_score_fr_100g",
                                      "product_name","sugars_100g", "saturated_fat_100g","code",
                                      "ingredients_text",  "ingredients_from_palm_oil_n", "ingredients_that_may_be_from_palm_oil_n"])

WGI_num0.columns

WGI_num0.describe()
```

```{code-cell} ipython3
pca = PCA()
WGI_num = pca.fit_transform(scale(WGI_num0))

pca.fit(WGI_num)
```

```{code-cell} ipython3

print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
```

```{code-cell} ipython3

eig = pd.DataFrame(
    {
        "Dimension" : ["Dim" + str(x + 1) for x in range(8)], 
        "Variance expliquée" : pca.explained_variance_,
        "% variance expliquée" : np.round(pca.explained_variance_ratio_ * 100),
        "% cum. var. expliquée" : np.round(np.cumsum(pca.explained_variance_ratio_) * 100)
    }
)
eig
```

```{code-cell} ipython3


eig.plot.bar(x = "Dimension", y = "% variance expliquée") # permet un diagramme en barres
plt.text(5, 18, "14%") # ajout de texte
plt.axhline(y = 14, linewidth = .5, color = "dimgray", linestyle = "--") # ligne 17 = 100 / 6 (nb dimensions)
plt.show()
```

```{code-cell} ipython3

WGI_pca = pca.transform(WGI_num)

# +
# Transformation en DataFrame pandas
WGI_pca_df = pd.DataFrame({
    "Dim1" : WGI_pca[:,0], 
    "Dim2" : WGI_pca[:,1],
    "Dim3" : WGI_pca[:,2],
    "Dim4" : WGI_pca[:,3],
    "Dim5" : WGI_pca[:,4],
    "product": data_study["product_name"],
    "nutrition_grade_fr" : data_study["nutrition_grade_fr"]
})
# Résultat (premières lignes)
WGI_pca_df.head()
```

```{code-cell} ipython3

WGI_pca_df.plot.scatter("Dim1", "Dim2", "Dim3") # nuage de points
plt.xlabel("Dimension 1 (35%)") # modification du nom de l'axe X
plt.ylabel("Dimension 2 (30%)") # idem pour axe Y
plt.suptitle("Premier plan factoriel (65%)") # titre général
plt.show()
```

```{code-cell} ipython3
WGI_num0
```

```{code-cell} ipython3
from yellowbrick.datasets import load_concrete
from yellowbrick.features import PCA
from yellowbrick.style import set_palette

# Load the concrete dataset
visualizer = PCA(scale=True, proj_features=True)
visualizer.fit_transform(WGI_num0)
visualizer.show()
```

```{code-cell} ipython3


WGI_num.shape

WGI_num
```

```{code-cell} ipython3

# -


n = WGI_num.shape[0] # nb individus
p = WGI_num.shape[1] # nb variables
eigval = (n-1) / n * pca.explained_variance_ # valeurs propres
sqrt_eigval = np.sqrt(eigval) # racine carrée des valeurs propres
corvar = np.zeros((p,p)) # matrice vide pour avoir les coordonnées
for k in range(p):
    corvar[:,k] = pca.components_[k,:] * sqrt_eigval[k]
# on modifie pour avoir un dataframe
coordvar = pd.DataFrame({'id': WGI_num0.columns, 'COR_1': corvar[:,0], 'COR_2': corvar[:,1], 'COR_3': corvar[:,2]
                        })
coordvar
```

```{code-cell} ipython3

WGI_num_t2 = data_study.drop(columns = ["nutrition_score_fr_100g", "energy_100g","main_category", 
                                      "product_name","sugars_100g", "saturated_fat_100g","code", 
                                      "ingredients_text","reg_energy","reg_additives",
                                     "carbohydrates_100g", "proteins_100g", "fat_100g",
                                     "fiber_100g", "salt_100g", "additives_n", "ingredients_from_palm_oil_n", "ingredients_that_may_be_from_palm_oil_n"])

WGI_num_t2.columns
```

```{code-cell} ipython3
WGI_num_t2["nutrition_grade_fr"].value_counts()
```

```{code-cell} ipython3
test_a=WGI_num_t2.loc[WGI_num_t2["nutrition_grade_fr"]=="A"].sample(1000)
test_b=WGI_num_t2.loc[WGI_num_t2["nutrition_grade_fr"]=="B"].sample(1000)
test_c=WGI_num_t2.loc[WGI_num_t2["nutrition_grade_fr"]=="C"].sample(1000)
test_d=WGI_num_t2.loc[WGI_num_t2["nutrition_grade_fr"]=="D"].sample(1000)
test_e=WGI_num_t2.loc[WGI_num_t2["nutrition_grade_fr"]=="E"].sample(472)

frames = [test_a, test_b, test_c, test_d, test_e]

test = pd.concat(frames)
```

```{code-cell} ipython3
del test["nutrition_grade_fr"]
```

```{code-cell} ipython3

#librairies pour la CAH
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
#générer la matrice des liens
Z = linkage(test,method='ward',metric='euclidean')
#affichage du dendrogramme
plt.title("CAH")
dendrogram(Z,labels=test.index,color_threshold=0)
plt.show()
```

```{code-cell} ipython3
WGI_num0.columns
```

```{code-cell} ipython3
WGI_num0.describe()
```

```{code-cell} ipython3

from sklearn.cluster import KMeans

kmeans2 = KMeans(n_clusters = 5)
kmeans2.fit(WGI_num0)
```

```{code-cell} ipython3

pd.Series(kmeans2.labels_).value_counts()
```

```{code-cell} ipython3

WGI_k2 = WGI_num0.assign(classe = kmeans2.labels_)
WGI_k2.groupby("classe").mean()

WGI_k2
```

```{code-cell} ipython3


WGI_pca_k2 = WGI_pca_df.assign(classe = kmeans2.labels_)
WGI_pca_k2.plot.scatter(x = "Dim1", y = "Dim2", c = "classe", cmap = "Accent")
plt.show()
```

```{code-cell} ipython3


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
```

```{code-cell} ipython3


inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters = k, init = "random", n_init = 20).fit(WGI_num0)
    inertia = inertia + [kmeans.inertia_]
inertia = pd.DataFrame({"k": range(1, 11), "inertia": inertia})
inertia.plot.line(x = "k", y = "inertia")
plt.scatter(2, inertia.query('k == 2')["inertia"], c = "red")
plt.scatter(5, inertia.query('k == 5')["inertia"], c = "red")
plt.show()
```

```{code-cell} ipython3
sns.heatmap(pd.crosstab(WGI_pca_k2.nutrition_grade_fr, WGI_pca_k2.classe, normalize = True))
```

```{code-cell} ipython3
t = pd.crosstab(WGI_pca_k2.nutrition_grade_fr, WGI_pca_k2.classe, normalize = "columns")
t = t.assign(nutrition_grade_fr = t.index)
tm = pd.melt(t, id_vars = "nutrition_grade_fr")
tm = tm.assign(value = 100 * tm.value)

sns.catplot("nutrition_grade_fr", y = "value", col = "classe", data = tm, kind = "bar")
```

```{code-cell} ipython3
tm
```

```{code-cell} ipython3
WGI_pca_k2
```

```{code-cell} ipython3
new_class = data_study.assign(classe = kmeans2.labels_)
```

```{code-cell} ipython3
new_class.describe()
```

Ces résultats sont assez proche du nutriscore. Il y a des poids à mettre en fonction des apports nutritionnels.
Il faut donc appliquer une règle métier.

```{code-cell} ipython3

```

# Combinaison des aliments

+++

Voici l'apport quotidien pour un adulte :

Glucides = 250g (carbohydrates)

lipides = 50 insaturé et 20 sature

Protéines = 45 g pour une personne de 55 kg 60 g pour une personne de 75 kg

Fibres = 25 à 30 g dont une moitié issue des céréales et l'autre issue des fruits et légumes

Mise à disposition d'une application Smart-Food.

Mise au panier d'aliment et calcul de l'apport quotidien Vert/rouge.

```{code-cell} ipython3

```

```{code-cell} ipython3

```
