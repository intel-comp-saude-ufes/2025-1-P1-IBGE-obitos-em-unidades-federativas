from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
from lib.colors import COLOR_RED, COLOR_RESET
from lib.utils import print_dict


def plot_clusters_pca(X_scaled, df_clusters, estado_index, metodo='AffinityPropagation', feature_names=None):
    """
    Visualiza os clusters via PCA 2D, imprime a variância explicada,
    e retorna as 2 features originais mais importantes para cada componente principal.

    Parâmetros:
    - X_scaled: array-like (normalizado), shape (n_amostras, n_features)
    - df_clusters: DataFrame com colunas de rótulos de clustering
    - estado_index: lista ou Index com nomes das amostras
    - metodo: string, coluna de df_clusters para plotar
    - feature_names: lista com nomes das colunas originais (opcional)

    Retorna:
    - dict com chaves 'PC1' e 'PC2', valores são listas com as 2 features originais mais importantes
    """

    if metodo not in df_clusters.columns:
        raise ValueError(f"O método '{metodo}' não foi encontrado em df_clusters.columns.")

    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X_scaled.shape[1])]

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_

    print(f"Variância explicada pelo PCA ({metodo}):")
    print(f"PC1: {var_exp[0]:.2%}")
    print(f"PC2: {var_exp[1]:.2%}")
    print(f"Total (2D): {var_exp.sum():.2%}\n")

    # Features mais importantes
    components = pca.components_
    important_features = {}
    for i, pc in enumerate(['PC1', 'PC2']):
        comp = components[i]
        top_indices = np.argsort(np.abs(comp))[-2:][::-1] ## isso inverte a lista
        top_features = [feature_names[idx] for idx in top_indices]
        important_features[pc] = top_features

    # DataFrame para plotagem
    df_plot = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': df_clusters[metodo].values,
        'Estado': estado_index
    })

    # Plot
    plt.figure(figsize=(10, 7))
    scatter = sns.scatterplot(
        x='PC1', y='PC2',
        hue='Cluster',
        data=df_plot,
        palette='tab10',
        s=100,
        edgecolor='black'
    )

    # Labels das amostras
    for i in range(len(df_plot)):
        plt.text(df_plot['PC1'][i] + 0.2, df_plot['PC2'][i], df_plot['Estado'][i], fontsize=8)

    # Legendas dos eixos com features
    plt.xlabel(f'PC1 ({var_exp[0]:.1%} var. explicada)')
    plt.ylabel(f'PC2 ({var_exp[1]:.1%} var. explicada)')

    plt.title(f'{metodo} - Agrupamento dos Estados (PCA)')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    return important_features


def plot_silhouette_scores(X_scaled, range_k=range(2, 8), img_path = '', title_suffix = ''):
    scores = []
    _k = 0
    for k in range_k:
        _k = k
        model = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
        score = silhouette_score(X_scaled, model.labels_)
        scores.append(score)

    plt.plot(list(range_k), scores, marker='o')
    plt.title(f"{title_suffix} - Silhouette Score por número de clusters")
    plt.xticks(list(range(0, _k)))
    plt.xlabel("k")
    plt.ylabel("Silhouette Score")

    if img_path == '' : plt.show()
    else : plt.savefig(img_path)
    
    plt.close()
        

def pca_analysis(data : np.array, feature_names, verbose = False, threshold = 0.9):
    
    num_components = 2
    for i in range(2, 10  + 1):
        if i == len(feature_names): break
        
        pca = PCA(n_components=i)
        _ = pca.fit_transform(data)
        var_exp = pca.explained_variance_ratio_
        
        num_components=i    
        if verbose:
            #print(f"PC1: {var_exp[0]:.2%}")
            #print(f"PC2: {var_exp[1]:.2%}")
            print(f"Total of explicability from a PCA({i}D): {var_exp.sum():.2%}\n")
            _flag = var_exp.sum() > threshold
            if _flag : break
                    
    # Features mais importantes
    components = pca.components_
    important_features = dict()
    
    for i, pc in enumerate([f'PC{_n}' for _n in range(1, num_components + 1)]):
        comp = components[i]
        top_idx = np.argsort(np.abs(comp))[-1] ## isso inverte a lista, [-2:][::-1]
        important_features[pc] = feature_names[top_idx]
    
    #print_dict(important_features)
    if verbose:
        for k, v in important_features.items():
            print(f'{k} : {v}')
    
    return important_features


## recebe um scaler do scikit
def apply_robust_scaling(data : pd.DataFrame) -> np.array:
    data_p = (RobustScaler()).fit_transform(data)
    return data_p