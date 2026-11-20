# Previsão do PIB dos Municípios da Região Sul — 2024

Trabalho 2 de **T326 – Ciência de Dados** (UNIFOR, Prof. Rilder de Sousa Pires).

## Equipe

- Matheus Diógenes Amorim — 2310277
- Lucas Kohagura Alencastro — 2320373

## Objetivo

Prever, com aprendizado de máquina, a economia de 2024 dos **1.191 municípios da Região Sul** (Paraná, Santa Catarina e Rio Grande do Sul), usando dados públicos do IBGE. São duas abordagens:

1. **Regressão:** prever o valor do PIB de 2024 de cada município (avaliada por R² e MAPE).
2. **Classificação:** prever a categoria de crescimento do PIB em 2024 em relação a 2023 (avaliada por F1-score e matriz de confusão):

| Categoria | Crescimento do PIB |
|---|---|
| Forte retração | menor que −5% |
| Retração | de −5% a −1% |
| Estabilidade | de −1% a 1% |
| Crescimento moderado | de 1% a 5% |
| Forte crescimento | acima de 5% |

Os modelos foram avaliados primeiro prevendo 2023 com dados até 2022 e comparando com o PIB oficial. Depois, foram retreinados com dados até 2023 para prever 2024.

Em resumo:
- **teste de 2023:** o erro médio do PIB previsto foi de cerca de 7% (MAPE);
- **2024:** a previsão é que o PIB da região cresça cerca de 6%.

**A metodologia, os modelos comparados, os resultados e as limitações estão no relatório técnico: [`reports/relatorio.pdf`](reports/relatorio.pdf).**

## Estrutura do repositório

```
notebooks/          passo a passo do projeto, em ordem:
  01_coleta_integracao.ipynb     coleta dos dados no IBGE e junção das fontes
  02_eda_preprocessamento.ipynb  análise exploratória e preparação dos dados
  03_regressao.ipynb             modelos de regressão (PIB)
  04_classificacao.ipynb         modelos de classificação (categoria de crescimento)
  05_previsao_2024.ipynb         previsões finais de 2024
src/                funções usadas pelos notebooks (coleta, preparação, modelos e gráficos)
data/raw/           dados brutos baixados do IBGE (gerados pelo notebook 01; fora do git)
data/processed/     dataset tratado e dicionário das variáveis (.csv)
data/predictions/   previsões de 2024 e do teste de 2023 (.csv)
reports/            relatório técnico, figuras e tabelas de resultados
executar.sh         roda o projeto inteiro de uma vez
requirements.txt    bibliotecas necessárias
```

## Como executar

Requisitos: **Python 3.12** e acesso à internet, porque o notebook 01 baixa os dados do IBGE.

```bash
git clone https://github.com/MatheuzinDev/Trabalho-Ciencia-Dados-AV2.git
cd Trabalho-Ciencia-Dados-AV2
```

> **Windows:** clone numa pasta de caminho curto (por exemplo `C:\projetos\`). Em pastas com caminho muito longo, a instalação das bibliotecas pode falhar por causa do limite de tamanho de caminho do Windows.

### Opção 1: tudo de uma vez (recomendado)

```bash
bash executar.sh
```

O script cria o ambiente virtual, instala as bibliotecas do `requirements.txt` e executa os 5 notebooks em ordem. O processo leva cerca de 30 minutos. O andamento aparece no terminal, e os resultados ficam salvos nos próprios notebooks e nas pastas `data/` e `reports/`.

No Windows, use o **Git Bash**, que vem com o [Git for Windows](https://git-scm.com/download/win).

Para rodar só alguns notebooks, informe os números. Por exemplo, para refazer só a modelagem, sem baixar os dados de novo:

```bash
bash executar.sh 03 04 05
```

### Opção 2: interativa, pelo Jupyter

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

O comando `jupyter notebook` só abre a interface no navegador. Na pasta `notebooks/`, abra cada notebook **em ordem, de `01` a `05`**, e use **Kernel → Restart & Run All** em cada um. Cada notebook usa os arquivos gerados pelo anterior.

### Reprodutibilidade

Os modelos usam semente aleatória fixa, e as versões das bibliotecas estão fixadas no `requirements.txt`. O projeto foi testado do zero, a partir de um clone limpo, e gerou exatamente os mesmos arquivos que estão no repositório.

Os dados foram coletados no IBGE em 08/11/2026. Se o IBGE revisar alguma série depois dessa data, uma nova coleta pode dar resultados um pouco diferentes.

## Dados

### Fontes

Todos os dados vêm do IBGE:
- [SIDRA](https://sidra.ibge.gov.br/): PIB dos Municípios, Produção Agrícola Municipal, Pesquisa da Pecuária Municipal, Estimativas de População, Pesquisa Industrial Anual e Pesquisa Anual de Serviços;
- [API de Localidades](https://servicodados.ibge.gov.br/api/docs/localidades): lista dos municípios da região.

As tabelas usadas de cada pesquisa estão no relatório.

### Instruções para download

Todos os arquivos estão em **.csv** (UTF-8, separados por vírgula). Valores em dinheiro estão em **mil reais**, como divulga o IBGE.

- **Tudo de uma vez:** `git clone` (acima) ou, no GitHub, **Code → Download ZIP**.
- **Um arquivo só:** abra o arquivo no GitHub e clique em **Download raw file**. Links diretos:
  - [`previsoes_pib_2024.csv`](https://raw.githubusercontent.com/MatheuzinDev/Trabalho-Ciencia-Dados-AV2/main/data/predictions/previsoes_pib_2024.csv)
  - [`previsoes_categoria_2024.csv`](https://raw.githubusercontent.com/MatheuzinDev/Trabalho-Ciencia-Dados-AV2/main/data/predictions/previsoes_categoria_2024.csv)
  - [`dataset_tratado.csv`](https://raw.githubusercontent.com/MatheuzinDev/Trabalho-Ciencia-Dados-AV2/main/data/processed/dataset_tratado.csv)
  - [`dicionario_dataset.csv`](https://raw.githubusercontent.com/MatheuzinDev/Trabalho-Ciencia-Dados-AV2/main/data/processed/dicionario_dataset.csv)
- **Direto no Python:**
  ```python
  import pandas as pd
  base = "https://raw.githubusercontent.com/MatheuzinDev/Trabalho-Ciencia-Dados-AV2/main/"
  previsoes = pd.read_csv(base + "data/predictions/previsoes_pib_2024.csv")
  ```
- **Dados brutos:** não ficam no repositório, porque somam cerca de 200 MB. Para baixá-los, rode o notebook 01.

### Dataset tratado (`data/processed/`)

| Arquivo | Conteúdo |
|---|---|
| `dataset_tratado.csv` | Dados usados para treinar os modelos. Cada linha é um município em um ano (2008 a 2024), com o PIB daquele ano e as variáveis usadas para prevê-lo. Em 2024 o PIB está vazio, porque é o que se quer prever |
| `dicionario_dataset.csv` | **Descrição de cada variável do dataset tratado:** significado, tipo, ano de referência e unidade |
| `painel_municipios.csv` e `dicionario_painel.csv` | Dados de todas as fontes juntos, antes da preparação, e a descrição das colunas |

**Variáveis-alvo** (o que os modelos aprendem a prever):

| Coluna | Descrição |
|---|---|
| `pib` | PIB do município no ano, em mil reais (regressão) |
| `cresc_pib_pct` | Crescimento do PIB em relação ao ano anterior, em % |
| `categoria_crescimento` | Categoria de crescimento, uma das 5 da tabela acima (classificação) |

As demais colunas são as variáveis explicativas: histórico do PIB, produção agrícola e pecuária, população, setores da economia e localização. A descrição de cada uma está no `dicionario_dataset.csv`, e os grupos estão explicados no relatório.

### Previsões (`data/predictions/`)

**`previsoes_pib_2024.csv`** traz o PIB previsto para 2024, com uma linha por município:

| Coluna | Descrição |
|---|---|
| `cod_ibge`, `municipio`, `uf` | Identificação do município |
| `pib_2023_oficial_mil_reais` | PIB oficial de 2023 |
| `pib_2024_previsto_mil_reais` | **PIB previsto para 2024** |
| `pib_2024_limite_inferior_80_mil_reais`, `pib_2024_limite_superior_80_mil_reais` | Faixa provável do PIB de 2024 (intervalo de 80%, calculado com os erros dos anos anteriores) |
| `crescimento_2024_previsto_pct` | Crescimento previsto em relação a 2023, em % |
| `modelo` | Modelo usado |

**`previsoes_categoria_2024.csv`** traz a categoria de crescimento prevista para 2024, com uma linha por município:

| Coluna | Descrição |
|---|---|
| `cod_ibge`, `municipio`, `uf` | Identificação do município |
| `categoria_2024_prevista` | **Categoria prevista para 2024** |
| `crescimento_2024_modelo_categoria_pct` | Crescimento previsto pelo modelo de classificação, em % |
| `categoria_pelo_pib_previsto` | Categoria calculada a partir do PIB previsto no outro arquivo. É dada só para comparação: as duas coincidem em 82% dos municípios |
| `modelo` | Modelo usado |

**Teste de 2023:** os arquivos `previsoes_2023_validacao.csv`, `regressao_teste_2023_todos_modelos.csv` e `classificacao_teste_2023_todos_modelos.csv` trazem as previsões de 2023 feitas com dados até 2022, ao lado dos valores oficiais. Os dois últimos incluem as previsões de todos os modelos comparados.
