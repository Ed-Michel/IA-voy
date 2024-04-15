import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

dataframe = pd.read_csv(r"C:\Users\edgar\Desktop\Inteligencia Artificial\IA\K-means\analisis.csv")
# print(dataframe.head())
# print(dataframe.describe())
# print(dataframe.groupby('categoria').size())

# dataframe.drop(['categoria'], axis=1, inplace=True)
# dataframe.hist()
# plt.show()

# Revisando la gráfica no pareciera que hay algún tipo de agrupación o correlación entre los usuarios y sus categorías.
# sb.pairplot(dataframe.dropna(), hue='categoria', height=4, vars=["op", "ex", "ag"], kind='scatter')
# plt.show()

# Se define la entrada
X = np.array(dataframe[["op", "ex", "ag"]])
y = np.array(dataframe['categoria'])
# (140, 3)
print(X.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Cada color representa una de las 9 categorías
colores = ['blue','red','green','blue','cyan','yellow','orange','black','pink','brown','purple']
asignar = [colores[row] for row in y]

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=asignar, s=60)

plt.show()

# Obtener el valor K (punto de codo)
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
plt.plot(Nc, score)
plt.xlabel('Numero de Clusters')
plt.ylabel('Puntuacion')
plt.title('Curva del Codo')
plt.grid(True)
plt.show()
# k = 5

# Se ejecuta el algoritmo K-Means
kmeans = KMeans(n_clusters=5).fit(X)
centroides = kmeans.cluster_centers_
print(centroides)

# Prediciendo los clusters
labels = kmeans.predict(X)

# Se obtienen los centroides de cada cluster
C = kmeans.cluster_centers_
colores = ['red','green','blue','cyan','yellow']
asignar = [colores[row] for row in labels]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=asignar, s=60)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=colores, s=1000)

plt.show()

# Se obtienen los valores y se grafican (op y ex)
f1 = dataframe['op'].values
f2 = dataframe['ex'].values

plt.scatter(f1, f2, c=asignar, s=70)
plt.scatter(C[:, 0], C[:, 1], marker='*', c=colores, s=1000)
plt.title('op y ex')
plt.show()

# Se obtienen los valores y se grafican (op y ag)
f1 = dataframe['op'].values
f2 = dataframe['ag'].values

plt.scatter(f1, f2, c=asignar, s=70)
plt.scatter(C[:, 0], C[:, 2], marker='*', c=colores, s=1000)
plt.title('op y ag')
plt.show()

# Se obtienen los valores y se grafican (ex y ag)
f1 = dataframe['ex'].values
f2 = dataframe['ag'].values

plt.scatter(f1, f2, c=asignar, s=70)
plt.scatter(C[:, 1], C[:, 2], marker='*', c=colores, s=1000)
plt.title('ex y ag')
plt.show()

# Determinar cuántos usuarios tiene cada cluster
copy = pd.DataFrame()
copy['usuario'] = dataframe['usuario'].values
copy['categoria'] = dataframe['categoria'].values
copy['label'] = labels;

cantidadGrupo = pd.DataFrame()
cantidadGrupo['color'] = colores
cantidadGrupo['cantidad'] = copy.groupby('label').size()
print(cantidadGrupo)

# Verificando la diversidad en las categorias
group_referrer_index = copy['label'] ==0
group_referrals = copy[group_referrer_index]

diversidadGrupo =  pd.DataFrame()
diversidadGrupo['categoria'] = [0,1,2,3,4,5,6,7,8,9]
diversidadGrupo['cantidad'] = group_referrals.groupby('categoria').size()
print(diversidadGrupo)

# Por ejemplo, se imprimen a los usuarios están más cerca del centroide por cada cluster
mas_cercano, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)

usuarios = dataframe['usuario'].values
for row in mas_cercano:
    print(usuarios[row])


# Por último, un ejemplo que predice al cluster que pertenece dependiendo de los valores de entrada asignados (op, ex y ag)
X_nuevo = np.array([[28.99, 39.31, 73.82]])

labels_otro = kmeans.predict(X_nuevo)
print("Pertenece al cluster: ", str(labels_otro))