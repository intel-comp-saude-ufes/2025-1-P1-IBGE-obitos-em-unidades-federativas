
import traceback
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import RobustScaler
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from matplotlib.patches import Rectangle
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from pathlib import Path

RESULTS='experiment_results/'

def plot_linkage(labels : str, dataset_scaled=None,  method='ward', year_of_dataset='2022'):
  linked = linkage(dataset_scaled, method=method)  # ou 'complete', 'average'
  plt.figure(figsize=(12, 6))
  dendrogram(linked, labels=labels, leaf_rotation=90)
  plt.title("Dendrograma - Agglomerative Clustering")
  plt.ylabel("Distância")
  plt.tight_layout()
  plt.savefig(os.path.join(RESULTS + year_of_dataset, f'agglomerative_{method}.pdf'), format='pdf')
  plt.close()


def plot_affinity_graph(X, labels, exemplars, index_labels=None, year_of_dataset='2022'):
    """
    Plota um grafo com conexões entre amostras e seus exemplares no Affinity Propagation.

    Parâmetros:
    - X: dados usados para clustering (não utilizados no plot, mas mantidos por consistência)
    - labels: vetor de rótulos de cluster para cada amostra
    - exemplars: índices dos centros de cluster (exemplares)
    - index_labels: lista ou Index com nomes das amostras (ex: df.index)
    """

    G = nx.Graph()

    if index_labels is None:
        index_labels = list(range(len(labels)))  # fallback para números

    # Mapear cada índice numérico para o nome real
    idx_to_name = dict(enumerate(index_labels))

    for i, label in enumerate(labels):
        G.add_node(i, label=label)

    # Conectar cada amostra ao seu exemplar
    exemplar_per_sample = exemplars[labels]
    for i, exemplar in enumerate(exemplar_per_sample):
        if i != exemplar:
            G.add_edge(i, exemplar)

    # Layout e cores
    pos = nx.spring_layout(G, seed=42)
    node_colors = [labels[i] for i in G.nodes]

    plt.figure(figsize=(12, 10))
    nx.draw(
        G, pos,
        labels={i: idx_to_name[i] for i in G.nodes},
        node_color=node_colors,
        cmap=plt.cm.tab10,
        with_labels=True,
        font_size=8,
        node_size=600,
        edge_color='gray'
    )
    plt.title("Affinity Propagation - Graph View (Samples → Exemplars)")
    plt.savefig(os.path.join(RESULTS + year_of_dataset, 'affinity_propagation.pdf'), format='pdf')
    plt.close()


def plot_silhouette_scores(X_scaled, range_k=range(2, 8), year_of_dataset='2022'):
    scores = []
    for k in range_k:
        model = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
        score = silhouette_score(X_scaled, model.labels_)
        scores.append(score)

    plt.plot(list(range_k), scores, marker='o')
    plt.title("Silhouette Score por número de clusters")
    plt.xlabel("k")
    plt.ylabel("Silhouette Score")
    plt.savefig(os.path.join(RESULTS + year_of_dataset, 'silhouette_scores.pdf'), format='pdf')
    plt.close()

def plot_clusters_pca(X_scaled, df_clusters, estado_index, metodo='AffinityPropagation', feature_names=None, year_of_dataset='2022'):
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

    #print(f"Variância explicada pelo PCA ({metodo}):")
    #print(f"PC1: {var_exp[0]:.2%}")
    #print(f"PC2: {var_exp[1]:.2%}")
    #print(f"Total (2D): {var_exp.sum():.2%}\n")

    # Features mais importantes
    components = pca.components_
    important_features = {}
    for i, pc in enumerate(['PC1', 'PC2']):
        comp = components[i]
        top_indices = np.argsort(np.abs(comp))[-2:][::-1]
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
    
    plt.savefig(os.path.join(RESULTS + year_of_dataset, f'{metodo}_cluster_pca.pdf'), format='pdf')
    plt.close()
    return important_features


def plot_heatmap_z_score(df=None, g=None, year_of_dataset='2022'):
    if not (g is None):
        df = df.loc[g, :].copy()

    z_df = (df - df.mean()) / df.std()

    plt.figure(figsize=(16, 10))
    ax = sns.heatmap(z_df, center=0, annot=True, linewidth=.5, cmap="vlag")

    ax.set(
        xlabel="Z-Score da Proporção de Óbitos por faixa de idade",
        ylabel="Estado"
    )
    ax.xaxis.tick_top()
    ax.tick_params(top=True, bottom=False)

    
    outliers_positive = set()
    outliers_negative = set()
    outlier_counts = {
        'outliers_positivos': [],
        'outliers_negativos': []
    }

    
    for col in z_df.columns:
        col_vals = z_df[col]
        count_pos = (col_vals > 2).sum()
        count_neg = (col_vals < -2).sum()
        outlier_counts['outliers_positivos'].append(count_pos)
        outlier_counts['outliers_negativos'].append(count_neg)

    # Heatmap e detecção
    for i in range(z_df.shape[0]):  # linhas (estados)
        for j in range(z_df.shape[1]):  # colunas (faixas de idade)
            val = z_df.iloc[i, j]
            if val > 2:
                ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=2.5))
                outliers_positive.add(z_df.index[i])
            elif val < -2:
                ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='blue', lw=2.5))
                outliers_negative.add(z_df.index[i])

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS + year_of_dataset, 'heatmap_z_score.pdf'), format='pdf')
    plt.close()

    # Construir DataFrame com contagem de outliers por coluna
    outlier_counts_df = pd.DataFrame(outlier_counts, index=z_df.columns)

    return {
        'outliers_positivos': sorted(outliers_positive),
        'outliers_negativos': sorted(outliers_negative)
    }, outlier_counts_df


def save_unique_figure(save_dir, filename_base, ext="pdf"):
    """
    Salva figura com nome único
    """
    i = 0
    while True:
        if i == 0:
            filename = f"{filename_base}_regression.{ext}"
        else:
            filename = f"{filename_base}_regression_{i}.{ext}"
        full_path = os.path.join(save_dir, filename)
        if not os.path.exists(full_path):
            break
        i += 1
    plt.savefig(full_path, format=ext)
    plt.close()


def regression(X=None, X_label=None, y=None, y_label=None, remove_top_n=None, remove_bottom_n=None, year_of_dataset='2022'):
    """
    Regressão linear simples com opção de remover os `n` maiores e/ou menores valores de y antes do ajuste.
    """
    # Garantir arrays numpy
    X = np.asarray(X)
    y = np.asarray(y)

    if remove_bottom_n == 0: remove_bottom_n = None
    if remove_top_n == 0: remove_top_n = None

    
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    
    assert X.shape[0] == y.shape[0], f"[ANTES DA MÁSCARA] X.shape={X.shape}, y.shape={y.shape}"

    
    mask = np.ones(len(y), dtype=bool)

    # Remover top n
    if remove_top_n is not None:
        if isinstance(remove_top_n, (int, float)) and remove_top_n > 0 and remove_top_n < len(y):
            top_idx = np.argsort(y)[-int(remove_top_n):]
            mask[top_idx] = False
        else:
            print(f"[WARNING] Valor inválido para remove_top_n: {remove_top_n}")

    # Remover bottom n
    if remove_bottom_n is not None:
        if isinstance(remove_bottom_n, (int, float)) and remove_bottom_n > 0 and remove_bottom_n < len(y):
            bottom_idx = np.argsort(y)[:int(remove_bottom_n)]
            mask[bottom_idx] = False
        else:
            print(f"[WARNING] Valor inválido para remove_bottom_n: {remove_bottom_n}")

    # Aplicar máscara
    X = X[mask].copy()
    y = y[mask].copy()

    # Verificação final
    assert X.shape[0] == y.shape[0], f"[DEPOIS DA MÁSCARA] Tamanhos incompatíveis: X={X.shape}, y={y.shape}"

    # Regressão
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)

    # Para o gráfico: garantir que X seja 1D
    if X.shape[1] > 1:
        X_plot = X[:, 0]
    else:
        X_plot = X.ravel()

    # Plotagem
    print(X_plot.shape, y.shape)
    plt.scatter(X_plot, y, color='blue', label='Data Points')
    plt.plot(X_plot, y_pred, color='red', label='Regression Line')
    plt.xlabel(X_label if X_label else "X")
    plt.ylabel(y_label if y_label else "y")
    plt.title('Linear Regression Model')
    plt.legend()
    plt.grid(True)

    # Adicionar R²
    plt.text(
        0.05, 0.95, f'$R^2$ = {reg.score(X, y):.3f}',
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray')
    )

    # Criar pasta, se não existir
    save_dir = os.path.join(RESULTS, year_of_dataset)
    os.makedirs(save_dir, exist_ok=True)

    safe_label = y_label.replace("/", "_") if y_label else "output"
    i = 0
    while True:
        if i == 0:
            filename = f'{safe_label}_regression.pdf'
        else:
            filename = f"{safe_label}_regression_{i}.pdf"
        full_path = os.path.join(save_dir, filename)
        if not os.path.exists(full_path):
            break
        i += 1

    plt.savefig(full_path, format='pdf')
    plt.close()


def plot_bar_sorted(df, column_name, highlight_states=None,
                    population=None,
                    highlight_color='red', default_color='skyblue',
                    title=None, figsize=(20,10),
                    year_of_dataset='2022'):
    """
    Plota gráfico de barras ordenado do menor para o maior valor, destacando alguns estados.
    Pode normalizar os valores pela população passada.

    Parâmetros:
    - df (pd.DataFrame): DataFrame com dados, índice sendo os estados.
    - column_name (str): Nome da coluna para valores.
    - highlight_states (list): Lista de estados a destacar (cor diferente).
    - population (pd.Series or list or np.array): População dos estados na mesma ordem do df.
    - highlight_color (str): Cor dos estados destacados.
    - default_color (str): Cor padrão das barras.
    - title (str): Título do gráfico.
    - figsize (tuple): Tamanho da figura.
    """
    df_copy = df.copy()
    
    if column_name not in df.columns:
        print(f'WARNING: {column_name} not in df.columns!')
        return

    # Se população foi passada, calcula valor per capita
    if population is not None:
        pop_series = population if isinstance(population, pd.Series) else pd.Series(population, index=df.index)
        df_copy['value_per_capita'] = df_copy[column_name] / pop_series
        y_col = 'value_per_capita'
        ylabel = f"{column_name} (proporção da população)"
        auto_title = f"{column_name} por proporção da população"
    else:
        y_col = column_name
        ylabel = column_name
        auto_title = f"{column_name} (valores absolutos)"

    # Ordenar pelo valor que será plotado (proporcional ou absoluto)
    df_sorted = df_copy.sort_values(by=y_col)

    colors = [highlight_color if state in (highlight_states or []) else default_color for state in df_sorted.index]

    plt.figure(figsize=figsize)
    plt.bar(df_sorted.index, df_sorted[y_col], color=colors)

    plt.xticks(rotation=45, ha='right')
    plt.ylabel(ylabel)
    plt.title(title or auto_title)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(RESULTS + year_of_dataset, f'{ylabel}_bar_sorted.pdf'), format='pdf')
    plt.close()