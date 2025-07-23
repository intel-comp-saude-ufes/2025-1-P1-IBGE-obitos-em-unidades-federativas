import traceback
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering

import pandas as pd
import os

from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import RobustScaler
import pandas as pd
from pathlib import Path

from lib.analysis import pca_analysis
from lib.data_processing import process_all_years
from lib.experiment_algorithms import RESULTS, plot_affinity_graph, plot_bar_sorted, plot_clusters_pca, plot_heatmap_outliers, plot_linkage, plot_silhouette_scores, regression




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

    Path(RESULTS + year_of_dataset).mkdir(exist_ok=True, parents=True)
    dataset2 = None

    try:
        dataset = pd.read_csv(os.path.join(path, f'result_{year_of_dataset} - done.csv')).set_index('Grandes Regiões').sort_index()
        dataset2 = dataset.copy()
        dataset = dataset.loc[:, [c for c in dataset.columns if '5.2 ' in c]].copy()
        dataset2 = dataset2.loc[:, list(set(dataset2.columns) - set(dataset.columns))].copy()
        dataset = dataset[sorted(dataset.columns)]
        dataset2 = dataset2[sorted(dataset2.columns)]
    except Exception as e:
        traceback.print_exc()
        exit(1)

    
    taxa_crescimento_geometrico = 1.0

    if year_of_dataset == '2023': 
        taxa_crescimento_geometrico = 1.0052  # crescimento de 0.52% em relação ao censo de 2022 | https://cidades.ibge.gov.br/brasil/pesquisa/10102/122229

    
    df_pop = pd.read_csv(os.path.join(path,'populacaoGrupoDeIdade2022.csv')).set_index('Brasil e Unidade da Federação').sort_index()
    df_pop = df_pop[sorted(df_pop.columns)]
    df_pop = df_pop * taxa_crescimento_geometrico
    df_pop_total = df_pop.loc[:, 'Total'].copy()
    df_pop_total.drop(index=['Brasil'], axis='rows', inplace=True)
    df_pop.drop(index=['Brasil'], axis='rows', inplace=True)
    
    
    dataset = dataset.astype(float)
   
    X = dataset.values
    P = df_pop.values 
    divisores = P[:, np.arange(X.shape[1]) % 8]
    X_norm = X / divisores
    dataset.iloc[:, :] = X_norm

    labels_clustering = dataset.index.tolist().copy()
    

    for method in ['ward', 'complete', 'average']:
        plot_linkage(labels_clustering, dataset_scaled=dataset, method=method, year_of_dataset=year_of_dataset) # 4 ou 5

    aff_model = AffinityPropagation(random_state=42)
    aff_labels = aff_model.fit_predict(dataset)

    plot_affinity_graph(
        X=dataset,
        labels=aff_model.labels_,
        exemplars=aff_model.cluster_centers_indices_,
        index_labels=dataset.index,
        year_of_dataset=year_of_dataset
    ) 


    plot_silhouette_scores(dataset, year_of_dataset=year_of_dataset)


    kmeans_model = KMeans(n_clusters=4 if year_of_dataset == '2022' else 5, random_state=42) 
    kmeans_labels = kmeans_model.fit_predict(dataset)

    agg_model = AgglomerativeClustering(n_clusters=4 if year_of_dataset == '2022' else 5, linkage='average') 
    agg_labels = agg_model.fit_predict(dataset)


    df_clusters = pd.DataFrame({
        'Estado': labels_clustering,
        'KMeans': kmeans_labels,
        'AffinityPropagation': aff_labels,
        'Agglomerative': agg_labels
    }).set_index('Estado')

    df_clusters.to_csv(RESULTS + '/' + year_of_dataset + '/resultado_agrupamentos.csv', index=True)


    f = None
    for i in range(len(df_clusters.columns)):
        f = plot_clusters_pca(X_scaled=dataset, 
                              df_clusters=df_clusters,
                            estado_index=labels_clustering, 
                            metodo=df_clusters.columns[i],
                            feature_names=dataset.columns,
                            year_of_dataset=year_of_dataset)
    
    
    

    subs = sorted(list(set([ c for _, v in f.items() for c in v ]))) # minimo 1 máximo 4


    with open(RESULTS + year_of_dataset + '/principais_causas_de_mortalidade.txt', 'w') as ff:
        ff.write('\n'.join(subs))

    

    cols = sorted(list(set([c for c in dataset.columns if any(sub in c for sub in subs)])))
    
    subset_dataset = dataset.loc[:, cols].copy()


    df_prop = subset_dataset
    df_prop.columns = df_prop.columns.str.extract(r'Grupos de idade_(.+)$')[0]

    plot_heatmap_outliers(df=subset_dataset, year_of_dataset=year_of_dataset) # outliers em relação ao estado

    faixas_de_idade =  [ faixa for faixa in df_pop.columns if any(faixa in s for s in subs) ]
    df_pop = df_pop.loc[:, faixas_de_idade]

    df_result = pd.concat([df_pop.add_suffix('_0'), df_pop.add_suffix('_1')], axis='columns') / df_pop_total.values.reshape(-1, 1)
   

    for j in range(len(df_result.columns)):
        X = df_result.iloc[:, j].values
        X_label = f'Proporção população | {df_result.columns[j]} / total do estado'
        y = df_prop.iloc[:, j].values
        y_label = df_prop.columns[j]
        regression(X, X_label, y, y_label, year_of_dataset)
    
    
    dataset2 = dataset2.dropna(axis=1, how='all')
    dataset2 = dataset2.astype(float)
    data = StandardScaler().fit_transform(dataset2.values)
    #dataset2.iloc[:, :] = data
    d2 = pca_analysis(data=data, feature_names=dataset2.columns)

    features = sorted(list(set([ v for _, v in d2.items()])))

    for i in range(len(features)):
        plot_bar_sorted(dataset2, features[i], year_of_dataset=year_of_dataset)
    