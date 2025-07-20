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

from lib.analysis import pca_analysis
from lib.data_processing import process_all_years


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


def run_experiments():
    years = list(map(lambda x : str(x), range(2022, 2023 + 1)))


    for year in years:
        filename = f'result_{year} - done.csv'
        if not os.path.exists('./'+filename):
            process_all_years(years, path='data/')
            break

    # Roda experimento para todos os anos
    for year in years:
        _run_experiment(year_of_dataset=year)
    

def _run_experiment(datasets_path='./', year_of_dataset='2022'):
    path = datasets_path


    ## Passo 1 - Ler o arquivo 

    # cria diretorio de resultados para esse ano, se não existir
    Path(RESULTS + year_of_dataset).mkdir(exist_ok=True, parents=True)
    dataset2 = None

    try:
        dataset = pd.read_csv(os.path.join(path, f'result_{year_of_dataset} - done.csv')).set_index('Grandes Regiões').sort_index()
        dataset2 = dataset.copy()
        dataset = dataset.loc[:, [c for c in dataset.columns if '5.2 ' in c]].copy()
        dataset2 = dataset2.loc[:, list(set(dataset2.columns) - set(dataset.columns))].copy()
    except Exception as e:
        traceback.print_exc()
        exit(1)
    

    # Passo 2 - Ajuste de escala , visualização de agrupamentos aglomerativo (dendograma) , affinity (grafo) e  kmeans (silhouette)  para definição do número de grupos

    scaler = RobustScaler()
    dataset_scaled = scaler.fit_transform(dataset)


    # visualização de agrupamento aglomerativo e affinity para definição do número de grupos

    for method in ['ward', 'complete', 'average']:
        plot_linkage(dataset.index.tolist(), dataset_scaled=dataset_scaled, method=method, year_of_dataset=year_of_dataset) # 4 ou 5

    aff_model = AffinityPropagation(random_state=42)
    aff_labels = aff_model.fit_predict(dataset_scaled)

    plot_affinity_graph(
        X=dataset_scaled,
        labels=aff_model.labels_,
        exemplars=aff_model.cluster_centers_indices_,
        index_labels=dataset.index
    ) 


    plot_silhouette_scores(dataset_scaled, year_of_dataset=year_of_dataset)


    # Passo 4 - Com um número de grupos definidos, fazer agrupamento e visualização com PCA(n_components=2), retornar as 2 principais features de cada componente


    kmeans_model = KMeans(n_clusters=4 if year_of_dataset == '2022' else 5, random_state=42) 
    kmeans_labels = kmeans_model.fit_predict(dataset_scaled)

    agg_model = AgglomerativeClustering(n_clusters=4 if year_of_dataset == '2022' else 5, linkage='average') 
    agg_labels = agg_model.fit_predict(dataset_scaled)


    df_clusters = pd.DataFrame({
        'Estado': dataset.index,
        'KMeans': kmeans_labels,
        'AffinityPropagation': aff_labels,
        'Agglomerative': agg_labels
    }).set_index('Estado')

    f = None
    for i in range(len(df_clusters.columns)):
        f = plot_clusters_pca(X_scaled=dataset_scaled, 
                              df_clusters=df_clusters,
                            estado_index=dataset.index, 
                            metodo=df_clusters.columns[i],
                            feature_names=dataset.columns,
                            year_of_dataset=year_of_dataset)
    
    
    

    subs = sorted(list(set([ c for _, v in f.items() for c in v ]))) # minimo 1 máximo 4


    with open(RESULTS + year_of_dataset + '/principais_causas_de_mortalidade.txt', '+a') as ff:
        ff.write('\n'.join(subs))

    cols = sorted(list(set([c for c in dataset.columns if any(sub in c for sub in subs)])))
    
    subset_dataset = dataset.loc[:, cols].copy()

    


    # Passo 5 - A partir do conjunto das principais causas de mortalidades por faixa de idade (1 a 4),
    # identificação de estados outliers em cada uma a partir de um heatmap 


    # faz sentido somente para 2022, vou deixar comentado


    taxa_crescimento_geometrico = 1.0

    if year_of_dataset == '2023': 
        taxa_crescimento_geometrico = 1.0052  # crescimento de 0.52% em relação ao censo de 2022 | https://cidades.ibge.gov.br/brasil/pesquisa/10102/122229

    
    df_pop = pd.read_csv(os.path.join(path,'populacaoGrupoDeIdade2022.csv')).set_index('Brasil e Unidade da Federação')
    
    faixas_de_idade =  ['Total'] + [ faixa for faixa in df_pop.columns if any(faixa in s for s in subs) ]

    df_pop = df_pop.loc[:, faixas_de_idade].copy()
    df_pop = df_pop * taxa_crescimento_geometrico

    df_pop_total = df_pop['Total'].drop(labels=['Brasil'],axis='rows').sort_index().copy()

    df_pop.drop(columns=['Total'],inplace=True)
    idx = sorted(list(set(df_pop.index) & set(subset_dataset.index)))
    df_pop = df_pop.loc[idx, :].copy()
    

    results = dict()

    for c in subset_dataset.columns:
        # Tenta extrair o grupo etário a partir do padrão no nome da coluna
        match = [col for col in df_pop.columns if f"Grupos de idade_{col}" in c]
        
        if len(match) == 1:
            idade = match[0]
            results[c] = (subset_dataset[c] / df_pop[idade]) * 100
        else:
            print(f"nenhuma ou múltiplas correspondências para a coluna: {c}")


    df_prop = pd.DataFrame( results , index=subset_dataset.index)

    #print(df_prop)

    d1 = plot_heatmap_z_score(df=df_prop) # outliers em relação ao estado

    #print(d1)

    outliers = [ s for _, v in d1[0].items() for s in v]
    
    # Atualiza os índices de d1[1] mantendo apenas a faixa etária
    d1[1].index = d1[1].index.str.extract(r'Grupos de idade_(.+)$')[0]
    df_prop.columns = df_prop.columns.str.extract(r'Grupos de idade_(.+)$')[0]
    
    

    df_prop = df_prop[sorted(df_prop.columns)]
    df_pop = df_pop[sorted(df_pop.columns)]

    #print(d1[1].info())
    #print(df_pop.info())
    #print(df_prop.info())
    

    #for i in range(len(df_prop.columns)):

    #    regression(X= (df_pop.iloc[:, i].values / df_pop_total.values).reshape(-1, 1),
    #         X_label = f'Proporção população | {df_prop.columns[i]} / total do estado',
    #         y=df_prop.iloc[:, i].values,
    #         y_label = 'y_label',
    #         remove_top_n=d1[1].iloc[i, 0],
    #         remove_bottom_n=d1[1].iloc[i, 1],
    #         year_of_dataset=year_of_dataset)
        
    faixas_idade = sorted(set(df_prop.columns) & set(df_pop.columns) & set(d1[1].index))

    def get_outlier_value(df, index, column):
        val = df.loc[index, column]
        if isinstance(val, pd.Series):
            if len(val) > 0:
                val = val.iloc[0]
            else:
                return None
        return int(val) if pd.notna(val) else None


    for idade in faixas_idade:
        i_prop = df_prop.columns.get_loc(idade)
        i_pop = df_pop.columns.get_loc(idade)
        
        shape = df_prop.iloc[:, i_prop].values.shape
        if len(shape) > 1:
            yy = df_prop.iloc[:, i_prop]
            
            for j in range(shape[1]):
                part = yy.iloc[:, j]
                regression(
                    X=(df_pop.iloc[:, i_pop].values / df_pop_total.values).reshape(-1, 1),
                    X_label=f'Proporção população | {idade} / total do estado',
                    y=part.values,
                    y_label=idade,
                    remove_top_n=get_outlier_value(d1[1], idade, 'outliers_positivos'),
                    remove_bottom_n=get_outlier_value(d1[1], idade, 'outliers_negativos'),
                    year_of_dataset=year_of_dataset
                )
        else:
            regression(
                X=(df_pop.iloc[:, i_pop].values / df_pop_total.values).reshape(-1, 1),
                X_label=f'Proporção população | {idade} / total do estado',
                y=df_prop.iloc[:, i_prop].values,
                y_label=idade,
                remove_top_n=get_outlier_value(d1[1], idade, 'outliers_positivos'),
                remove_bottom_n=get_outlier_value(d1[1], idade, 'outliers_negativos'),
                year_of_dataset=year_of_dataset
            )

        
    dataset2 = dataset2.dropna(axis=1, how='all') ## columns that are copies of another ones but with somehow all entries equal to zero.
    d2 = pca_analysis(data=dataset2.values, feature_names=dataset2.columns)

    features = sorted(list(set([ v for _, v in d2.items()])))

    for i in range(len(features)):
        plot_bar_sorted(dataset2, features[i], population=df_pop_total, highlight_states= outliers, year_of_dataset=year_of_dataset)
    
