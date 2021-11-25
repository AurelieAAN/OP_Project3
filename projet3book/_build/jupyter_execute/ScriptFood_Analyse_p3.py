#!/usr/bin/env python
# coding: utf-8

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

# In[1]:


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
