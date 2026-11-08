# Previsão do PIB dos Municípios da Região Sul — 2024

Trabalho 2 de **T326 – Ciência de Dados** (UNIFOR, Prof. Rilder de Sousa Pires).

Modelos de aprendizado de máquina para prever, para os **1.191 municípios da Região Sul** (PR, SC e RS):

1. **Regressão:** o PIB de 2024 (métricas: R² e MAPE)
2. **Classificação:** a categoria de crescimento do PIB em 2024 (métricas: F1-score e matriz de confusão)

| Categoria | Crescimento do PIB em relação ao ano anterior |
|---|---|
| Forte retração | < -5% |
| Retração | -5% a -1% |
| Estabilidade | -1% a 1% |
| Crescimento moderado | 1% a 5% |
| Forte crescimento | > 5% |

Protocolo: treinar com dados até 2022 e validar contra o PIB oficial de 2023; depois retreinar com dados até 2023 e prever 2024.

## Estrutura

```
data/raw/          dados brutos do SIDRA (gerados pela coleta, fora do git)
data/processed/    dataset tratado + dicionário de variáveis (.csv)
data/predictions/  previsões de 2023 (validação) e de 2024 (.csv)
notebooks/         01 coleta → 02 EDA → 03 regressão → 04 classificação → 05 previsão 2024
src/               funções reutilizadas pelos notebooks
reports/           relatório técnico (PDF) e figuras
```

## Como executar

Requisitos: Python 3.12.

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Rode os notebooks em ordem, de `01` a `05`. O notebook `01` baixa todos os dados direto da API do SIDRA/IBGE (precisa de internet).

## Dados

Fontes (SIDRA/IBGE): PIB dos Municípios, Produção Agrícola Municipal, Pesquisa da Pecuária Municipal, Pesquisa Industrial Anual – Empresa, Pesquisa Anual de Serviços e Estimativas de População.

Os arquivos tratados e as previsões ficam em `data/processed/` e `data/predictions/`. *(Instruções de download e descrição das variáveis: a completar na Fase 8.)*

## Equipe

- Matheus Diógenes Amorim — 2310277
- Lucas Kohagura Alencastro — 2320373
