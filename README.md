# Indicadores Sociais e Óbitos por Faixa Etária no Brasil (2022–2023): Uma Análise de Dados do IBGE com Técnicas de Mineração de Dados
## Objetivo
Este trabalho se propõe a aplicação de métodos de mineração de dados bem estabelecidos na literatura, sobre indicadores sociais e mortalidades por diferentes causas selecionadas e/ou faixas etárias nas unidades federativas brasileiras, com base em dados tabulares oficiais do [Instituto Brasileiro de Geografia e Estatística (IBGE)](https://www.ibge.gov.br/).

## O que o repositório possui
Este repositório contém todo o código, dados e resultados referentes ao estudo sobre indicadores sociais e mortalidade por faixa etária no Brasil entre 2022 e 2023. A seguir, uma explicação resumida do que o repositório contém: 

- Dados brutos e tratados organizados em pastas, com arquivos Excel e CSV.

- Scripts em Python para pré-processamento, análise estatística, aplicação de técnicas de mineração de dados, regressão e agrupamentos (clustering).

- Resultados dos experimentos já gerados, como gráficos, relatórios e tabelas, organizados por ano e faixa etária.

- Documentação detalhada para reproduzir os experimentos e entender o fluxo do projeto.

- Link para o [artigo científico](./2025-1-P1-IBGE-obitos-em-unidades-federativas.pdf) referente à pesquisa que gerou o repositório e link para o resumo explicativo do artigo em [vídeo](https://youtu.be/9gbirT5KC_A).

Este material foi desenvolvido para facilitar a reprodução dos experimentos e servir de base para futuras pesquisas em análise de dados em saúde pública.


---
# Base de dados
Os dados obtidos para esse trabalho foram coletados a partir da base de dados do IBGE, especificamente, todas as 147 tabelas da base [Síntese de Indicadores Sociais](https://www.ibge.gov.br/estatisticas/sociais/saude/9221-sintese-de-indicadores-sociais.html) e dados do [censo demográfico de 2022](https://sidra.ibge.gov.br/tabela/9514). Esses dados foram filtrados e refinados para melhorar na qualidade dos resultados dos experimentos. Todos os arquivos processados dessas fontes estão disponíveis no diretório `data/`.


---
# Estrutura do repositório
Este projeto é estruturado para:
1. Armazenar **dados brutos e tratados**.
2. Executar **modelos e análises estatísticas** com scripts organizados.
3. Produzir e armazenar **gráficos e resultados prontos para o artigo**.

## Diretórios

### `data/`
Contém os **dados brutos e processados** utilizados no projeto, organizados em subpastas por temas (`1-uteis` a `5-uteis`) e por tipo de arquivo (Excel e CSV).
- Tabelas de indicadores sociais, econômicos, de saneamento, educação, saúde, etc.
- Subpastas `csv/`: versões dos dados já convertidos para `.csv`.

### `experiment_results/`
Resultados dos experimentos, organizados por ano (`2022/`, `2023/`):
- Regressões, agrupamentos (Affinity Propagation, Agglomerative, KMeans).
- Gráficos em PDF.
- Relatórios de métricas e análises.

### `lib/`
Biblioteca interna com os módulos do projeto:
- `data_processing.py`: Pré-processamento dos dados.
- `analysis.py`: Análises dos dados.
- `experiments.py`: Scripts de experimentos.
- `experiment_algorithms.py`: Algoritmos utilizados.
- `utils.py`: Funções auxiliares.
- `colors.py`: Definições de cores para visualização.

### `results/`
Resultados intermediários e finais das análises, agrupados por faixas de dados ou por ano.

---

# Como usar

Para reproduzir os resultados apresentados neste código, é necessário ter o **Python** instalado em seu sistema, bem como o **pip** — o gerenciador de pacotes do Python. Além disso, você deve ter o código disponível localmente na sua máquina. O código do repositório pode ser obtido de duas formas principais:

### 🔧 Opção 1: Clonar via Git
```bash
git clone https://github.com/intel-comp-saude-ufes/2025-1-P1-IBGE-obitos-em-unidades-federativas.git
cd 2025-1-P1-IBGE-obitos-em-unidades-federativas
```

### 📄 Opção 2: Baixar ZIP
1. Acesse o repositório no GitHub.
2. Clique no botão verde **"Code"**.
3. Selecione **"Download ZIP"**.
4. Extraia o arquivo ZIP em uma pasta da sua preferência.
5. Abra o terminal (ou prompt de comando) dentro da pasta onde extraiu os arquivos para executar os comandos e scripts.

### Executando o código
A seguir, apresentamos as instruções recomendadas para reproduzir os experimentos e gerar os resultados.

1. Instalar dependências do projeto:
```bash
pip install -r requirements.txt
```
2. Executar o projeto:
```
python3 main.py
```
3. Verifique o diretório `experiment_results/` para verificar o resultado dos experimentos.

---
# Licença
Este projeto está licenciado sob os termos da [Licença MIT](./LICENSE).

---
# Contato
| Autor                 | GitHub               | E-mail               |
| :---------------- | :------: | ----: |
| Pedro Igor Gomes de Morais | [@Pedro2um](https://github.com/Pedro2um) | pedro.i.morais@edu.ufes.br |
| Matheus Saick De Martin | [@saick123](https://github.com/saick123) | matheus.martin@edu.ufes.br |
| Renzo Henrique Guzzo Leão | [@Renzo-Henrique](https://github.com/seuusuario) | renzolealguzzo@gmail.com |

