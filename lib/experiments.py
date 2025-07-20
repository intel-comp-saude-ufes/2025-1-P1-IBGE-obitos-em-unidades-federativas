import traceback
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering

import pandas as pd
import os

from sklearn.preprocessing import RobustScaler
import pandas as pd
from pathlib import Path

from lib.analysis import pca_analysis
from lib.data_processing import process_all_years
from lib.experiment_algorithms import RESULTS, plot_affinity_graph, plot_bar_sorted, plot_clusters_pca, plot_heatmap_z_score, plot_linkage, plot_silhouette_scores, regression




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
        dataset = dataset[sorted(dataset.columns)]
        dataset2 = dataset2[sorted(dataset2.columns)]
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
        index_labels=dataset.index,
        year_of_dataset=year_of_dataset
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

    d1 = plot_heatmap_z_score(df=df_prop, year_of_dataset=year_of_dataset) # outliers em relação ao estado

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
    
