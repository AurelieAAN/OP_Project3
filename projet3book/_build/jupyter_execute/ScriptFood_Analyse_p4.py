#!/usr/bin/env python
# coding: utf-8

# # IV - Objectif une meilleure santé pour tous : Application Smart Food

# Un classement des aliments et le calcul de leur combinaison permettront d'améliorer la santé des français et les guider vers une alimentation équilibrée sans se priver.

# ## 1) Classement des aliments

# Etudions de plus près nos aliments et regardons qu'elles sont les variables qui permettent d'expliquer au mieux cette population.
# 
# 
# Pour commencer, il nous faut exclure les données manquantes et étudier les variables quantitatives qui sont corrélées.

# In[1]:


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
