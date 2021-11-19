#!/usr/bin/env python
# coding: utf-8
# %%

# # Présentation générale du jeu de données

# Installation des différents packages :

# pip install -r requirements.txt

# Chargement des librairies suivantes qu'on utilisera pour l'analyse de données :

# %%


import pandas as pd
import numpy as np
import missingno
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import folium
import json


# %%

def data_transformcol_string(data):
    """
    Mettre toutes les données en majuscule
    
    
    
    Args:
        data ([type]): données
    
    
    Returns:
        [type]: données
    
    """
    for col in data.select_dtypes(include=['object']).columns:
        if type(data[col].loc[pd.isna(data[col])==False].iloc[0])== str:
            try:
                data[col]=data[col].astype('|S')
            except:
                print("error - caractères spéciaux : "+col)
                continue
                
def data_fillNA_string(data):
    """
    Mettre toutes les données en majuscule
    
    
    
    Args:
        data ([type]): données
    
    
    Returns:
        [type]: données
    
    """
    for col in data.select_dtypes(include=['object']).columns:
        if type(data[col].loc[pd.isna(data[col])==False].iloc[0])== str:
            try:
                data[col].fillna("N.A", inplace=True)
            except:
                print("error - update na : "+col)
                continue
                
def data_uniqueone_string(data):
    """
    Mettre toutes les données en majuscule
    
    
    
    Args:
        data ([type]): données
    
    
    Returns:
        [type]: données
    
    """
    result=data
    for col in data.columns:
        if data[col].nunique()<= 1:
            print("deleted unique colonne : "+col)
            result=data.drop(col, axis=1,inplace=True)
        else:
            print(col+" - count unique : "+str(data[col].nunique()))
    return result

# %%

def data_missingTab(data):
    """
    Mettre toutes les données en majuscule
    
    
    
    Args:
        data ([type]): données
    
    
    Returns:
        [type]: données
    
    """
    percent_missing = data.isnull().sum() * 100 / len(data)
    missing_value_df = pd.DataFrame({'column_name': data.columns,
                                 'percent_missing': round(percent_missing,2)})
    
    return missing_value_df


# %%

def data_majuscule(data):
    """
    Mettre toutes les données en majuscule
    
    
    
    Args:
        data ([type]): données
    
    
    Returns:
        [type]: données
    
    """
    result = data.applymap(lambda s:s.upper() if type(s) == str else s)
    return result


# %%


def del_Nan(data, seuil, delete, all_freq):
    """
    Détection des données manquantes en fonction d'un seuil
    
    
    
    Args:
        data ([type]): données
        seuil ([type]): entre 0 et 1
        delete ([type]): 0 ou 1
        all_freq ([type]): resultat de toutes les colonnes
    
    Exemple:
    del_Nan(data, 0.8,0,0)
    
    """
    tab = pd.isnull(data).sum()/len(data)
    s=tab.to_frame(name='freq').reset_index()
    if all_freq==0:
        return s[s['freq']>=seuil]
    if all_freq==1:
        return s
    if delete==1:
        s=s[s['freq']>=seuil]
        col_del=list(s['index'])
        data=data.drop(list(s['index']), axis=1, inplace=True)
        print("Columns deleted : ", col_del)
# %%
#créons une fonction
def matrix_vm(data_i, fig_i, color_i):
    """
    Graphique Matrice des données manques
    
    
    
    Args:
        data_i ([type]): données
        fig_i ([type]): taille du graphique
        color_i ([type]): couleur
    
    Exemple:
    matrix_vm(data, (14,8), (0.564, 0.823, 0.631))
    
    """
    missingno.matrix(data_i, figsize=fig_i, fontsize=12, color=color_i, sparkline=False).set_title("Matrice des données manquantes")
    gray_patch = mpatches.Patch(color=color_i, label='Données présentes')
    white_patch = mpatches.Patch(color='white', label='Données manquantes')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', handles=[gray_patch, white_patch])


# %%

def graph_stripplot(data_i,column_i, title_i, scale_i, color_i):
    """
    Nuage de point d'une variable quantitative (volume de données important)
    
    
    
    Args:
        data_i ([type]): données
        column_i ([type]): colonne
        title_i ([type]): titre
        scale_i ([type]): taille du graphique
    
    
    
    """
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=scale_i)
    ax = sns.stripplot(y=column_i, data=data_i,
                       size=4, color=color_i).set_title(title_i)




# %%



def graph_boxenplot(data_i, column_i, color_i,title_i,scale_i):
    """
    boxenplot : plus quantile que le boxplot (même principe: valeur de lettre)
    
    
    
    Args:
        data_i ([type]): data
        column_i ([type]): colonne
        color_i ([type]): couleur
        title_i ([type]): titre
        scale_i ([type]): taille du graphique
    
    Exemple:
    graph_boxenplot(data, "circonference_cm", "g","Boxenplot de la circonférence en cm des arbres",(8,6))
    
    
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=scale_i)
    sns.boxenplot(x=column_i, 
                  color=color_i,
                  scale="linear", data=data_i)
    plt.title(title_i)



# %%


def outliers(data, var, delete_i):
    """
    Détection des valeurs aberrantes.
    Construit un tableau avec :
    - l'écart-interquartile de la valeur haute et basse
    - le nombre et le % d'outliers détectés
    
    
    Args:
        data : données
        s ([type]): données d'une colonne
        delete_i ([type]): 1/0 : yes/no
    
    Exemple:
    outliers(data.circonference_cm)
    
    
    """
    s= var.loc[pd.isna(var)==False]
    iqr = (np.quantile(s, 0.75))-(np.quantile(s, 0.25))
    upper_bound = np.quantile(s, 0.75)+(1.5*iqr)
    print(np.quantile(s, 0.25))
    lower_bound = np.quantile(s, 0.25)-(1.5*iqr)
    f = []
    for i in s:
        if i > upper_bound:
            f.append(i)
        elif i < lower_bound:
            f.append(i)
    sums = len(f)
    pros = len(f)/len(s)*100
    d = {'IQR':iqr,
         'Upper Bound':upper_bound,
        'Lower Bound':lower_bound,
        'Sum outliers': sums,'percentage outliers':pros}
    d = pd.DataFrame(d.items(),columns = ['sub','values'])
    if delete_i==1:
        data=data.loc[(var <= upper_bound) | pd.isna(var)==True]
        data=data.loc[(var >= lower_bound) | pd.isna(var)==True]
        print("data deleted => upper bound and lower bound")
        return data
    else:
        return(d)

# %%

def delete_outliers_UPPER(data_i, s, value_i):
    """
    Suppression des valeurs aberrantes supérieures à une valeur.
    
    
    Args:
        data_i ([type]): toutes les données
        s ([type]): données d'une colonne
        value ([type]): valeur maximum
    
    Exemple:
    outliers(data.circonference_cm, 700)
    
    
    """
    data_i=data_i.loc[(s <= value_i) | pd.isna(s)==True]
    return data_i
    
def delete_outliers_LOWER(data_i, s, value_i):
    """
    Suppression des valeurs aberrantes supérieures à une valeur.
    
    
    Args:
        data_i ([type]): toutes les données
        s ([type]): données d'une colonne
        value ([type]): valeur minimum
    
    Exemple:
    outliers(data.circonference_cm, 0)
    
    
    """
    data_i=data_i.loc[(s >= value_i) | pd.isna(s)==True]
    return data_i


def graph_hist(var, bins_r, title_i,color_h, xmin,xmax, xscale, ymin, ymax, xlabel, ylabel, scale_i):
    """
    Création d'un histogramme
    
    
    
    Args:
        var ([type]): données de la colonne
        bins_r ([type]): plage de l'axe x
        title_i ([type]): titre
        color_h ([type]): couleur
        xmin ([type]): min de l'axe x
        xmax ([type]): max de l'axe x
        xscale ([type]): taille du graphique
        ymin ([type]): min de l'axe y
        ymax ([type]): max de l'axe y
        xlabel ([type]): nom de l'axe x
        ylabel ([type]): nom de l'axe y
        scale_i ([type]): taille du graphique
    
    Exemple:
    graph_hist(data.circonference_cm,[0,50,100,150,200,250,350,450,700], "Répartition des arbres en fonction de leur circonference en cm", (0.564, 0.823, 0.631),
          0,700, 150, 0, 70000, 'circonference_cm', 'Fréquences',(11,7))
    
    """
    plt.figure(figsize=scale_i)
    plt.hist(var,bins=bins_r, color=color_h)
    plt.title(title_i)
    plt.xlim(xmin, xmax, xscale)
    plt.ylim(ymin,ymax)

    # Etiquetage
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


# %%
def graph_hist_sans_intervalles(var, title_i,color_h, xmin,xmax, xscale, ymin, ymax, xlabel, ylabel, scale_i):
    """
    Création d'un histogramme
    
    
    
    Args:
        var ([type]): données de la colonne
        bins_r ([type]): plage de l'axe x
        title_i ([type]): titre
        color_h ([type]): couleur
        xmin ([type]): min de l'axe x
        xmax ([type]): max de l'axe x
        xscale ([type]): taille du graphique
        ymin ([type]): min de l'axe y
        ymax ([type]): max de l'axe y
        xlabel ([type]): nom de l'axe x
        ylabel ([type]): nom de l'axe y
        scale_i ([type]): taille du graphique
    
    Exemple:
    graph_hist(data.circonference_cm,[0,50,100,150,200,250,350,450,700], "Répartition des arbres en fonction de leur circonference en cm", (0.564, 0.823, 0.631),
          0,700, 150, 0, 70000, 'circonference_cm', 'Fréquences',(11,7))
    
    """
    plt.figure(figsize=scale_i)
    plt.hist(var,color=color_h)
    plt.title(title_i)
    plt.xlim(xmin, xmax, xscale)
    plt.ylim(ymin,ymax)

    # Etiquetage
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


# %%


def graph_barplot(data, title, color_i,ylim_min, ylim_max, xlabel_i, ylabel_i, rotate_i, graph_vertical, fig_i):
    """
    Création d'un diagramme en barre
    
    
    
    Args:
        data ([type]): données
        title ([type]): titre
        color_i ([type]): couleur
        ylim_min ([type]): min axe y
        ylim_max ([type]): max axe y
        xlabel_i ([type]): nom de l'axe x
        ylabel_i ([type]): nom de l'axe y
        rotate_i ([type]): rotation des noms de l'axe x
        graph_vertical ([type]): diagramme en barre vertical = 1
        fig_i ([type]): taille du graphique

    Exemple:
    graph_barplot(data['reg_circonference_cm'], "Répartition des arbres en fonction des circonferences en cm (regroupés en intervalle)", 
                  (0.564, 0.823, 0.631),
              0, 40, "Intervalle - circonference_cm", "Fréquence en %",70, 1, (14,8))
    """
    t = pd.crosstab(data, "freq", normalize=True)
    t = t.assign(var = t.index, freq = 100 * t.freq)
    plt.figure(figsize=fig_i)
    if graph_vertical==1:
        sns.barplot(x = "var", y = "freq", data = t, color=color_i)
        plt.ylim(ylim_min,ylim_max)
    else:
        sns.barplot(x = "freq", y = "var", data = t, color=color_i)
        plt.xlim(ylim_min,ylim_max)
    plt.title(title)
    plt.xlabel(xlabel_i)
    plt.ylabel(ylabel_i)
    plt.xticks(rotation=rotate_i)
        


# %%




def graph_violinplot(data_col_i, title_i, ylabel_i, color_i,fig_i):
    """
    Création d'un violinplot :

    
    Args:
        data_col_i ([type]): données d'une colonne
        title_i ([type]): titre
        ylabel_i ([type]): nom de l'axe des y 
        color_i ([type]): couleur
        fig_i ([type]): taille du graphique
    
    Exemple: 
    graph_violinplot(data["hauteur_cm"], "Violinplot - Variable hauteur_cm", "hauteur_cm", (0.564, 0.823, 0.631), (14,7))
    
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=fig_i)
    ax = sns.violinplot(y=data_col_i, color=color_i)
    ax.set_title(title_i)
    ax.set_ylabel(ylabel_i)


# %%




# %%

def graph_boxplot(data_i, column_i, title_i, color_i, fig_i):
    """
    Boite à moustache
    
    
    
    Args:
        data_i ([type]): données
        column_i ([type]): colonne
        title_i ([type]): titre
        color_i ([type]): couleur
        fig_i ([type]): taille
    
    Exemple:
    graph_boxplot(data, "circonference_cm", "test", "g", (14,8))
    
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=fig_i)
    sns.boxplot(y=column_i, color=color_i, data=data_i)
    plt.title(title_i)

# %%

def graph_boxplot_by_group(data_i, column_i, groupby_i, title_i, color_i, fig_i):
    """
    Boite à moustache par catégorie
    
    
    
    Args:
        data_i ([type]): données
        column_i ([type]): colonne
        by_i ([type]): colonne qualitative
        title_i ([type]): titre
        color_i ([type]): couleur
        fig_i ([type]): taille
    
    Exemple:
    graph_boxplot_by(data, "circonference_cm","category", "test", "g", (14,8))
    
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=fig_i)
    sns.boxplot(y=column_i,x=groupby_i, palette=color_i, data=data_i)
    plt.title(title_i)

# %%

def graph_bubbleplot(data_i, color_i,xlabel_i, ylabel_i, title_i, ylim_min, ylim_max,
                    color_text, align, fontweight_i, scale_i):
    """
    Grphique bubbleplot
    
    
    
    Args:
        data_i ([type]): données de la colonne à représenter
        color_i ([type]): couleur des bubbles
        xlabel_i ([type]): nom de l'axe x 
        ylabel_i ([type]): nom de l'axe y
        title_i ([type]): titre
        ylim_min ([type]): min de l'axe y
        ylim_max ([type]): max de l'axe y
        color_text ([type]): couleur du texte
        align ([type]): alignement du texte
        fontweight_i ([type]): style du texte
        scale_i ([type]): Taille du graphique    
    
    Exemple:
    graph_bubbleplot(data['domanialite'], 'g',"Domanialite", "Fréquence en %", 
                 "Répartition des arbres par catégorie des domanialites", 0, 90,
                    "black", "center", "bold", (8,5))
    
    """
    t = data_i
    data_var=t.value_counts(normalize=True).reset_index(name="values")
    data_var['values']=100*data_var['values']
    plt.figure(figsize=scale_i)
    sns.scatterplot(data=data_var, x='index', y='values', s=100*data_var['values'], color=color_i).margins(0.4)
    plt.xlabel(xlabel_i)
    plt.ylabel(ylabel_i)
    plt.title(title_i)
    plt.ylim(ylim_min,ylim_max)
    for i,txt in enumerate(data_var['values']):
        plt.text(i,txt, str(round(txt,2))+" %", color=color_text, ha=align, fontweight=fontweight_i, size=12)


# %%


def graph_circle(data, column, title):
    """
    Graphique circulaire
    
    
    
    Args:
        data ([type]): données de la colonne à representer
        column ([type]): nom de la colonne
        title ([type]): titre du graphique
    
    Exemple:
    graph_circle(data["libelle_francais"], "libelle_francais", "Répartition des arbres en fonction de leur appellation")
    
    """
    t = pd.crosstab(data, "freq", normalize=True)
    t = t.assign(column = t.index, freq = 100 * t.freq)
    plt.figure(figsize=(18,8))
    plt.pie(t.freq, labels = t.column,  autopct='%.0f%%')

    plt.title(title)
    my_circle = plt.Circle((0,0), 0.4, color='white')
    p = plt.gcf()
    p.gca().add_artist(my_circle)


# Réalisons un diagramme circulaire représentant nos données de la variable libelle_francais.

# %%


def graph_pie(data, column, title, colors_i, fig_i):
    """
    graphique de type camembert
    
    
    
    Args:
        data ([type]): données de la colonne à représenter
        column ([type]): colonne
        title ([type]): titre
        colors_i ([type]): couleur
    
    Exemple:
    graph_pie(data["remarquable"], "remarquable", "Répartition des arbres selon s'ils sont, ou non, remarquable",['#8FD1A0', '#46A964'])
    
    """
    t = pd.crosstab(data, "freq", normalize=True)
    t = t.assign(column = t.index, freq = 100 * t.freq)
    color_palette_list = colors_i
    plt.figure(figsize=fig_i)
    plt.pie(t.freq, labels = t.column,  autopct='%.2f%%', colors=color_palette_list)
    plt.title(title)
    


# %%





# Dans notre jeu de données, nous savons dans quel arrondissement se trouve les arbres. Nous pouvons par exemple illustrer le nombre d'arbres que nous avons par arrondissement.
# 
# Pour cela, nous pouvons récupérer les données geographiques des arrondissements qui se trouve sur le site de Paris.
# source : https://opendata.paris.fr/explore/dataset/arrondissements/export/?disjunctive.c_ar&disjunctive.c_arinsee&disjunctive.l_ar&location=13,48.85156,2.32327

# %%

def carte_paris_arr(data_i, columns_i,  title_i, fill_color_i):
    """
    Carte de Paris : Visualiser une colonne en fonction des arrondissements
    
    
    
    Args:
        data_i ([type]): [description]
        columns_i ([type]): [description]
        title_i ([type]): [description]
        fill_color_i ([type]): [description]
    
    Exemple:
    carte_paris_arr(data, ["arrondissement_num", "hauteur_cm"], 'Représentation graphique de la hauteur (en cm) des arbres par arrondissement de Paris', "YlOrRd")
    
    """
    geo = json.load(open("arrondissements.geojson"))

    paris = folium.Map(location = [48.856578, 2.351828], zoom_start = 12)
    loc = title_i
    title_html = '''
                <h3 align="center" style="font-size:16px"><b>{}</b></h3>
                '''.format(loc) 
    folium.Choropleth(geo_data = geo, key_on = "feature.properties.c_ar",
                    data = data, columns = columns_i,
                    fill_color = fill_color_i).add_to(paris)
    paris.get_root().html.add_child(folium.Element(title_html))
    paris


# Nous remarquons que les arbres les plus haut se trouve au nord-ouest de Paris.

# Il serait intéressant de savoir aussi où se trouve les arbres les plus "gros".

# %%

# %%


def plot_quanti(data_i, column_x, column_y, scale_i, color_i, title_i, ylabel_i, xlabel_i): 
    """
    [summary]
    
    
    
    Args:
        data_i ([type]): données
        column_x ([type]): nom de la colonne sur l'axe x
        column_y ([type]): nom de la colonne sur l'axe y
        scale_i ([type]): taille du graphique
        color_i ([type]): couleur
        title_i ([type]): titre
        ylabel_i ([type]): nom de l'axe y
        xlabel_i ([type]): nom de l'axe x
    
    Exemple:
    
    plot_quanti(data, "circonference_cm", "hauteur_cm", (8,5),
                "g", "Nuage de point entre la circonference (cm) et la hauteur (cm) des arbres",
            "hauteur_cm", "circonference_cm")
    
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=scale_i)
    ax=sns.relplot(x=column_x, y=column_y, data=data_i, color=color_i, marker="+", linewidth=1);
    plt.title(title_i)
    plt.ylabel(ylabel_i)
    plt.xlabel(xlabel_i)


# %%



