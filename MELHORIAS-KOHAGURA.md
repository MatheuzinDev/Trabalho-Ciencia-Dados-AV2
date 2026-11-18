# Revisão do relatório: correções e melhorias

**Para:** Lucas Kohagura Alencastro
**Arquivo revisado:** `Overleaf_Ciencia_de_Dados.pdf` (versão de 17/11/2026, 11 páginas)
**Comparado com:** o enunciado do Trabalho 2 e os resultados deste repositório (`reports/resultados/`, `data/predictions/` e os notebooks 01 a 05)

## Resumo

A estrutura do relatório está boa. Todas as seções exigidas existem, e ele tem 11 páginas e 5 figuras, dentro dos limites de 15 páginas e 5 figuras. Algumas partes batem com o repositório:

- o painel de 27.393 × 105;
- as fontes do SIDRA;
- os limites das categorias;
- a estrutura setorial com 3 anos de atraso.

**O problema é a modelagem e os resultados (seções 1.1 e 4 a 9).** Elas descrevem modelos, arquivos e números que **não existem no repositório**. O professor vai abrir o GitHub, e os números do relatório precisam ser exatamente os que o código produz.

A prioridade é a seção 1. As seções 2 e 3 vêm depois.

---

## 1. Correções obrigatórias: números e modelos que não batem com o repositório

| Item | Está no relatório | O certo, segundo o repositório | Onde conferir |
|---|---|---|---|
| Modelos de regressão | Ridge, Random Forest, HistGradientBoosting | Ridge, Huber, Random Forest, LightGBM e **Ensemble** (média de Ridge, RF e LightGBM) | notebook 03; `src/modelos.py` |
| Regressão escolhida | Random Forest | **Ensemble (Ridge + RF + LightGBM)**, treinado com alvos a partir de **2016** | `config_regressao.json` |
| MAPE da validação (2019–2022) | 11,99% | **7,18%** | `regressao_validacao_comparacao.csv` |
| Teste de 2023 da regressão escolhida | R² 0,9956; MAPE 9,38% | **R² 0,9952; MAPE 6,96%** | `regressao_teste_2023.csv` |
| Persistência (PIB_t = PIB_t-1) | MAPE 12,92% | 12,92% (está certo) | `regressao_teste_2023.csv` |
| Classificadores | Logística, RF, HistGradientBoosting | Logística, Random Forest, LightGBM e **Regressão → categoria** (o crescimento previsto pelo Ensemble convertido nas 5 categorias) | notebook 04 |
| Classificador escolhido | RF, F1 macro 0,2455 | Etapa A: **RF** (F1 macro de 0,347 na validação e 0,275 no teste de 2023). Etapa B: **Regressão → categoria** (0,354 na validação 2019–2023) | `classificacao_*.csv` |
| Logística no teste | F1 macro 0,129 | **0,221** | `classificacao_teste_2023.csv` |
| Número de variáveis | 32 | **42**: 39 numéricas e 3 categóricas (`uf`, `regiao_intermediaria`, `porte`) | `src/features.py`, função `colunas_modelo` |
| Nomes de arquivos | `dataset_temporal.csv`, `dicionario_temporal.csv`, `metricas_por_classe_2023.csv`, coluna `categoria_classificador_independente` | `dataset_tratado.csv`, `dicionario_dataset.csv`, `classificacao_teste_2023.csv`. As colunas de categoria são `categoria_2024_prevista` e `categoria_pelo_pib_previsto` | `data/processed/`, `data/predictions/` |
| Hiperparâmetros na Etapa B | "mantidos da Etapa A" | **Reescolhidos com validação 2019–2023**, pelo mesmo critério. Na regressão só o LightGBM mudou (perda L1, `reg_lambda` = 5). Na classificação o escolhido passou a ser a Regressão → categoria | `config_*_etapa_b.json` |
| Crescimento do PIB em 2024 | Sul +10,71%; PR +10,32%; RS +12,62%; SC +8,80% | **Sul +6,08%** (de R$ 1,83 tri para R$ 1,95 tri); **PR +3,14%; RS +7,69%; SC +7,88%** | `previsao_2024_agregado_uf.csv` |
| Categorias de 2024 | 98,8% "Forte crescimento" e nenhuma retração | ver a tabela abaixo | `previsoes_categoria_2024.csv` |
| Incerteza | "previsões pontuais, sem intervalos" | **Há um intervalo empírico de 80%**, calculado com os erros da validação 2019–2023 e separado por porte do município. Ele está nas colunas `pib_2024_limite_inferior_80_mil_reais` e `pib_2024_limite_superior_80_mil_reais` | notebook 05, seção 2 |
| Concordância entre a categoria do PIB e a do classificador | 88,83% | **82,4%** com a mesma categoria, 16,5% em categoria vizinha e 1,2% com 2 ou mais categorias de diferença | notebook 05, seção 3 |
| Tabela dos municípios extremos (pág. 9) | São Pedro das Missões +43%, Jardim Olinda +3% | ver a seção 3.4 | `previsoes_pib_2024.csv` |

**Categorias de 2024 corretas** (`previsoes_categoria_2024.csv`, coluna `categoria_2024_prevista`):

| Categoria | Municípios | % |
|---|---:|---:|
| Forte retração | 201 | 16,9 |
| Retração | 131 | 11,0 |
| Estabilidade | 83 | 7,0 |
| Crescimento moderado | 194 | 16,3 |
| Forte crescimento | 582 | 48,9 |

**Categorias por estado em 2024** (`previsao_2024_categorias_uf.csv`):

- PR: 41,6% em "Forte retração" e só 9,8% em "Forte crescimento";
- RS: 85,3% em "Forte crescimento";
- SC: 40,3% em "Forte crescimento" e 27,8% em "Crescimento moderado".

> O resultado atual do relatório, com 98,8% em "Forte crescimento" e nenhuma retração, é o ponto mais fácil de ser questionado. Ele contradiz a queda da soja no Paraná, que o nosso modelo capta.

### Tabelas completas para copiar

**Regressão: teste de 2023 (Etapa A, modelos treinados até 2022)**

| Modelo | R² | MAPE (%) |
|---|---:|---:|
| Huber | 0,9937 | 6,59 |
| **Ensemble (Ridge + RF + LightGBM)**, o escolhido | **0,9952** | **6,96** |
| Random Forest | 0,9963 | 7,05 |
| LightGBM | 0,9952 | 7,24 |
| Ridge | 0,9932 | 7,29 |
| Baseline: crescimento da UF em t-1 | 0,9927 | 9,65 |
| Baseline: média de 3 anos do município | 0,9895 | 10,62 |
| Baseline: persistência | 0,9894 | 12,92 |

> O Huber foi o melhor no teste, mas um dos piores na validação (9,99%). Por isso **a escolha não muda depois do teste**: o modelo é escolhido só pela validação. Vale escrever isso, porque mostra que não houve vazamento do teste para a escolha.

**Classificação: teste de 2023 (Etapa A)**

| Modelo | F1 macro | F1 ponderado | Acurácia | Acerto exato ou na categoria vizinha |
|---|---:|---:|---:|---:|
| Regressão → categoria (Ensemble) | 0,358 | 0,706 | 0,651 | 0,868 |
| LightGBM (clf) | 0,304 | 0,635 | 0,566 | 0,802 |
| **Random Forest (clf)**, o escolhido na Etapa A | 0,275 | 0,604 | 0,530 | 0,796 |
| Regressão Logística | 0,221 | 0,459 | 0,354 | 0,501 |
| Baseline: classe majoritária | 0,176 | 0,691 | 0,785 | 0,860 |
| Baseline: categoria do ano anterior | 0,172 | 0,480 | 0,407 | 0,554 |

**Escolha da Etapa B (validação 2019–2023):**

| Abordagem | Validação | Resultado |
|---|---|---|
| Regressão | MAPE médio por modelo | **Ensemble @2016: 7,16%**; LightGBM 7,44%; RF 7,47%; Ridge 8,45%; Huber 9,31% |
| Classificação | F1 macro médio por modelo | **Regressão → categoria @2013: 0,354**; LightGBM 0,335; RF 0,334; Logística 0,298 |

Esses números vêm de `regressao_validacao_etapa_b.csv` e `classificacao_validacao_etapa_b.csv`.

---

## 2. O relatório contra o enunciado

| Requisito do enunciado | Situação | O que fazer |
|---|---|---|
| No máximo 15 páginas e 5 figuras | ✅ 11 páginas e 5 figuras | As figuras 1 e 2 juntam dois gráficos cada. Prefira 5 figuras de um gráfico só (ver a seção 3.6) |
| Introdução: problema e objetivo | ⚠️ curta demais | Dizer que a Região Sul foi a sorteada, por que prever o PIB municipal é útil e qual o contexto (estiagem de 2022 no RS, peso do agro) |
| Dataset: fontes, coleta, municípios, variáveis | ✅ | Só corrigir os nomes dos arquivos e o número de variáveis |
| Análise exploratória: análises, ausentes, duplicados, transformações, seleção | ⚠️ fraca | Ver a seção 3.1 |
| Modelagem: modelos, validação, procedimentos | ❌ descreve outros modelos | Reescrever com os modelos do repositório (seção 1) |
| Resultados: métricas, comparação, discussão, PIB e categorias de 2024 | ❌ números errados e discussão rasa | Seções 1, 3.2 e 3.4 |
| Conclusão: resultados, limitações, melhorias | ⚠️ falta a principal limitação | Incluir as enchentes do RS em maio de 2024 (seção 3.5) |
| Contribuição dos autores | ⚠️ imprecisa | Ver a seção 3.7 |
| Referências | ⚠️ incompletas | Ver a seção 3.8 |

---

## 3. Melhorias de conteúdo

### 3.1 Análise exploratória mais forte

Hoje a análise mostra só um histograma e duas correlações. Sugestões:

- **Choque agrícola × crescimento do PIB.** É o principal achado do trabalho. Use a figura `eda_relacoes_choque_reversao.png` e os números do notebook 02: a correlação de Spearman entre o choque agrícola da região imediata e o crescimento do PIB foi de **0,54**.
- **Desbalanceamento das categorias ao longo dos anos** (`fig_categorias_por_ano.png`). Mostra que "Forte crescimento" domina quase todos os anos (78,5% em 2023). Isso justifica o uso do F1 macro.
- O enunciado pede explicitamente mapas de calor. Já existe `fig_heatmap_correlacoes.png`. Se não couber como figura, cite-o em uma frase e descreva o que ele mostra.

### 3.2 Interpretação do modelo (critério C3 da avaliação)

O relatório não diz **quais variáveis importam**. Use a importância por permutação (`regressao_importancia_permutacao_2023.csv` e `reg_importancia_permutacao.png`). O número indica quanto o MAPE piora quando a variável é embaralhada:

| Variável | Quanto o MAPE piora |
|---|---|
| `choque_pam` (choque agrícola do município) | **+1,96 ponto** |
| `choque_soja` | +0,39 ponto |
| `var_pam_valor` | +0,29 ponto |
| `cresc_pib_l2` | +0,13 ponto |
| `choque_pam_uf` | +0,11 ponto |

A mensagem é que o PIB dos municípios do Sul é guiado pela safra.

### 3.3 Justificar as métricas e as dificuldades

- **Por que o F1 macro.** A classe majoritária acerta 78,5% em 2023, mais que qualquer modelo, mas tem F1 macro de só 0,18, porque nunca acerta as outras classes.
- **Por que "Estabilidade" quase não é acertada.** A faixa tem só 2 pontos de largura (-1% a 1%), enquanto o erro típico do crescimento previsto é de 5 a 7 pontos.
- **O RF da Etapa A previu retração demais em 2023:** 227 municípios em "Forte retração", contra 85 reais. O treino terminava em 2022, ano de estiagem, e os pesos balanceados empurram o modelo para as classes raras. Isso explica por que a Etapa B escolheu a Regressão → categoria.
- **Explicar a nova escolha da Etapa B.** Em 2024 o ano de 2023 já é conhecido. Por isso a escolha foi refeita com 5 anos de validação, com a mesma regra e definida antes de rodar.

### 3.4 Análise das previsões de 2024

- **O PR tem duas leituras.** O agregado cresce +3,1%, mas o **município mediano cai -2,5%**. As grandes cidades, menos dependentes da lavoura, sustentam o total.
- **Maiores altas previstas**, todas nas Missões (RS), onde o valor das lavouras supera o próprio PIB:

  | Município | UF | Crescimento previsto |
  |---|---|---:|
  | Itacurubi | RS | +103,8% |
  | Bossoroca | RS | +102,9% |
  | Capão do Cipó | RS | +100,5% |
  | Jóia | RS | +99,4% |

- **Maiores quedas previstas**, todas no norte e noroeste do PR, com forte queda no valor da soja:

  | Município | UF | Crescimento previsto |
  |---|---|---:|
  | Leópolis | PR | -34,4% |
  | Quinta do Sol | PR | -29,1% |
  | Janiópolis | PR | -27,7% |
  | Mariluz | PR | -26,0% |

- **Intervalo de 80%** por porte do município:

  | Porte | Intervalo |
  |---|---|
  | Até 10 mil habitantes | de -11,6% a +11,8% |
  | De 10 a 50 mil | de -9,0% a +10,9% |
  | Acima de 50 mil | de -6,7% a +9,7% |

As conclusões do notebook 05 trazem esse texto já organizado.

### 3.5 Limitações que faltam

- **Enchentes de maio de 2024 no RS.** Nenhuma fonte usada registra o evento. Municípios muito atingidos foram previstos em "Forte crescimento": Eldorado do Sul (+11,3%), Roca Sales (+8,2%) e Muçum (+6,9%). Provavelmente estão superestimados. **É a principal limitação das previsões de 2024 e precisa estar no relatório.**
- **Viés visto em 2023.** O modelo subestimou o crescimento geral em cerca de 3 pontos e previu 729 municípios em "Forte crescimento", contra 935 reais. Se o mesmo acontecer em 2024, a participação de "Forte crescimento" pode estar subestimada.
- **Os extremos são os menos confiáveis.** As previsões de ±30% a +100% dependem quase só do choque agrícola, e os municípios pequenos são os que mais erram.

### 3.6 Sugestão para as 5 figuras

Todas já estão em `reports/figures/`, prontas:

1. `eda_relacoes_choque_reversao.png`: análise exploratória (choque agrícola e reversão do crescimento)
2. `reg_validacao_mape_por_ano.png`: comparação dos modelos na validação temporal
3. `fig_regressao_previsto_real_2023.png`: PIB previsto × real no teste de 2023
4. `fig_matriz_confusao_2023.png`: matriz de confusão do teste de 2023
5. `fig_categorias_2024.png`: categorias previstas para 2024 por estado

Se quiser trocar alguma, as outras opções são `reg_importancia_permutacao.png`, `clf_validacao_f1_por_ano.png` e `fig_categorias_por_ano.png`.

### 3.7 Contribuição dos autores

O texto "Matheus: criação do repositório" não descreve o que foi feito. Sugestão:

- **Matheus Diógenes Amorim:** coleta e integração dos dados (SIDRA e API de Localidades), análise exploratória e pré-processamento, criação das variáveis, modelagem de regressão e de classificação, validação temporal, previsões de 2024 e organização do repositório.
- **Lucas Kohagura Alencastro:** redação do relatório técnico e análise dos resultados.

### 3.8 Referências

Acrescentar:

- **LightGBM:** Ke, G. et al. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS, 2017.
- **scikit-learn:** Pedregosa, F. et al. *Scikit-learn: Machine Learning in Python*. JMLR 12, 2011.
- **pandas:** o projeto pandas (citar a versão do `requirements.txt`).
- **IBGE:** nota metodológica do PIB dos Municípios.

---

## 4. Formatação (Overleaf)

- **Tabelas sem número nem título.** Use `\caption{...}` dentro de `table`, para ficar "Tabela 1 – ...", como já foi feito nas figuras.
- **Fórmulas em texto simples** ("R2 = 1 - soma((y - previsão)2)..."). Use modo matemático:
  - `$R^2 = 1 - \frac{\sum (y_i - \hat y_i)^2}{\sum (y_i - \bar y)^2}$`
  - `$\text{MAPE} = \frac{100}{n}\sum \left|\frac{\hat y_i - y_i}{y_i}\right|$`
  - `$g = \log(PIB_t / PIB_{t-1})$`
- **Defeitos de texto:**
  - pág. 7: "R2e MAPE", sem espaço;
  - pág. 5: o "R²" quebrou de linha;
  - nomes como `PIB_t-1` aparecem com sublinhado literal; use `$PIB_{t-1}$`.
- **Tabela da pág. 9:** não diz o que mostra. Coloque o título "Os 4 menores e os 4 maiores crescimentos previstos para 2024".
- **Seção 1.1 (Resumo dos resultados):** depois de corrigida, deve mostrar o Ensemble (MAPE 6,96% no teste) e a Regressão → categoria (F1 macro 0,358 no teste de 2023).

---

## 5. Entrega

- O PDF final deve ser salvo como **`reports/relatorio.pdf`**. É o único PDF que o `.gitignore` deixa entrar no repositório.
- Antes de gerar a versão final, confira cada número contra os arquivos de `reports/resultados/` e `data/predictions/`, ou contra as conclusões no fim dos notebooks 03, 04 e 05.

## Checklist

- [ ] Reescrever as seções 1.1 e 4 a 9 com os modelos e os números do repositório (seção 1)
- [ ] Trocar os nomes de arquivos e de colunas pelos reais
- [ ] Incluir a nova escolha da Etapa B e o intervalo de 80%
- [ ] Reforçar a análise exploratória (choque agrícola e desbalanceamento das classes)
- [ ] Incluir a importância das variáveis
- [ ] Incluir a limitação das enchentes do RS em 2024
- [ ] Ajustar as 5 figuras
- [ ] Corrigir a contribuição dos autores e as referências
- [ ] Numerar as tabelas e passar as fórmulas para LaTeX
- [ ] Salvar a versão final em `reports/relatorio.pdf`
