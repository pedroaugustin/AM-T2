import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing

# Define o cluster (classe) para cada ponto das instancias se baseando na menor distancia euclidiana entre os centroids
def define_cluster(centroids, k_clusters):
    dist = np.linalg.norm(instances - centroids[0,:], axis=1).reshape(-1,1)
    
    for x in range(1,k_clusters):
        dist = np.append(dist,np.linalg.norm(instances - centroids[x,:],axis=1).reshape(-1,1),axis=1)

    clusters = np.argmin(dist,axis=1)

    return clusters

# Recalcula os novos centroids fazendo a média entre todas instancias do cluster
def calcula_centroids(clusters,centroids):
    for classe in set(clusters):
        centroids[classe,:] = np.mean(instances[clusters == classe,:],axis=0)

    return centroids

# Realiza o k-means 
# Repete as funções acima 100 vezes
def k_means(centroids, k_clusters):
    num_iteracoes = 100

    for x in range(num_iteracoes):
        clusters = define_cluster(centroids, k_clusters)
        new_centroids = calcula_centroids(clusters, centroids)
        centroids = new_centroids
    
    return clusters, new_centroids

# Plota o gráfico dos clusters e centroids encontrados
def grafico_clusters(new_centroids, clusters, k_clusters):
    for x in range(k_clusters):
        plt.scatter(instances[clusters == x, 0], instances[clusters == x, 1], s = 1, c=[rgb[x]])
    plt.scatter(new_centroids[:,0],new_centroids[:,1], s=50, c = 'r',marker = '+')

    return

# Plota o gráfico dos clusters e centroids encontrados para o bank_t2
def grafico_clusters_bank(new_centroids, clusters):
    plt.scatter(instances[clusters == 0, 0], instances[clusters == 0, 1], s = 3, c='b')
    plt.scatter(instances[clusters == 1, 0], instances[clusters == 1, 1], s = 3, c='g')
    plt.scatter(new_centroids[:,0],new_centroids[:,1], s=50, c = 'r',marker = '+')

    return

# Calcula o wss para o elbow method
# Soma das distancias entre todas instancias associadas a um determinado cluster e o seu respectivo centroide
def wss(new_centroids, clusters):
    euc_distance = []
    for classe in set(clusters):
        euc_distance.extend((np.linalg.norm(instances[clusters == classe,:] - new_centroids[classe,:], axis=1))**2)

    soma = np.sum(euc_distance)

    return soma

# Realiza o k-means várias vezes (número dado por range_clusters) e calcula o wss para cada número de clusters
def find_elbow(range_clusters):
    eixoY = []
    eixoX = np.arange(1,range_clusters + 1)    

    for x in range(1,range_clusters + 1):
        k_clusters = x
        centroids = np.random.randint(1000000, size=(k_clusters,2))
        clusters, new_centroids = k_means(centroids, k_clusters)
        eixoY.append(wss(new_centroids, clusters))

    return eixoX, eixoY

# Plota o gráfico do wss pelo número de clusters
def grafico_wss(eixoX, eixoY):
    plt.plot(eixoX,eixoY,marker="o")
    plt.xticks(eixoX)
    plt.xlabel("Número de clusters")
    plt.ylabel("WCSS")
    plt.title("WCSS por número de clusters")

    return

# Realiza a transformação dos atributos categóricos do dataset bank_t2 para atributos numéricos
# Foi utilizado o label encoding
# Realiza a normalização dos valores numéricos do dataset
def bank_t2(atributo1, atributo2):
    df = pd.read_csv('bank_t2.txt',sep='\t')
    atributos_cat = ['marital', 'education', 'contact', 'day.of.week', 'poutcome']
    df_norm = df.copy()

    # Scikit utilizado para normalizar os valores e fazer o label encoding
    scaler = preprocessing.MinMaxScaler()
    le = preprocessing.LabelEncoder()

    df_norm[['age','duration.contact', 'number.contatcs.campaign']] = scaler.fit_transform(df_norm[['age','duration.contact', 'number.contatcs.campaign']])
    for atributo in atributos_cat:
        le.fit(df_norm[atributo])
        df_norm[atributo] = le.transform(df_norm[atributo])
    arrayfinal = np.vstack((df_norm[atributo1], df_norm[atributo2])).T

    return arrayfinal

# Inicialização das variáveis
rgb = []
menor_index = 15
range_clusters = 15

# Variável para escolher o dataset. 
# 0 = test_instances
# 1 = bank_t2
dataset = 0

# Atributos para o gráfico do bank_t2
# age marital education	housing.loan personal.loan contact day.of.week duration.contact	number.contatcs.campaign poutcome 
atributo1 = 'age'
atributo2 = 'marital'

# Inicia as instancias dependendo do dataset
if dataset == 0:
    instances = np.loadtxt('test_instances.txt')
    ground_truth = np.loadtxt('test_centroids.txt')
    k_clusters = 15
elif dataset == 1:
    instances = bank_t2(atributo1, atributo2)
    k_clusters = 2

# Cores para os gráficos de clusters
for x in range(k_clusters):
    rgb.append(np.random.rand(3,)) 

if dataset == 0:
    # Executa 100 vezes o k-means e encontra os centroids com melhor centroid_index
    for x in range(100):
        # Executa o k-means
        centroids = np.random.randint(1000000, size=(k_clusters,2))
        clusters, new_centroids = k_means(centroids,k_clusters)

        # Calcula distância entre os centroids encontrados e os centroids do ground_truth
        distancia = np.linalg.norm(ground_truth - new_centroids[0,:], axis=1).reshape(-1,1)
        for x in range(1,k_clusters):
            distancia = np.append(distancia,np.linalg.norm(ground_truth - new_centroids[x,:],axis=1).reshape(-1,1),axis=1)

        mais_proximos = np.argmin(distancia,axis=0)
        centroid_index = k_clusters - np.unique(mais_proximos).size

        if centroid_index < menor_index:
            melhores_centroids = new_centroids
            menor_index = centroid_index

    print(melhores_centroids)

    # Plota os gráficos
    plt.figure(1)
    plt.title("Centróides e clusters k-means")
    grafico_clusters(melhores_centroids, clusters, k_clusters)

    plt.figure(2)
    for x in range(k_clusters):
        plt.scatter(instances[:,0], instances[:,1], s = 1, c='b')
    plt.title("Centróides e clusters ground truth")
    plt.scatter(ground_truth[:,0],ground_truth[:,1], s=50, c = 'r',marker = '+')

    # Elbow method
    plt.figure(3)
    eixoX, eixoY = find_elbow(range_clusters)
    grafico_wss(eixoX, eixoY)
    plt.show()

elif dataset == 1:
    centroids = np.random.randint(1, size=(k_clusters,2))
    clusters, new_centroids = k_means(centroids,k_clusters)
    grafico_clusters_bank(new_centroids, clusters)
    plt.xlabel(atributo1)
    plt.ylabel(atributo2)
    plt.show()






