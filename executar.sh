#!/usr/bin/env bash
# Executa o projeto inteiro de uma vez: cria o ambiente virtual, instala as
# dependências e roda os notebooks 01 a 05 em ordem ("Restart & Run All" de cada um).
#
# Uso (na raiz do repositório; no Windows, pelo Git Bash):
#   bash executar.sh            # roda os 5 notebooks (~30 min; o 01 baixa os dados do IBGE)
#   bash executar.sh 03 04 05   # roda só os notebooks indicados (ex.: só a modelagem)
#
# Cada notebook usa os arquivos gravados pelo anterior, então o script para no
# primeiro que falhar. As saídas ficam gravadas dentro dos próprios notebooks,
# e o tempo de cada etapa vai para o arquivo executar.log.
set -u

RAIZ="$(cd "$(dirname "$0")" && pwd)"
cd "$RAIZ" || exit 1
LOG="$RAIZ/executar.log"
: > "$LOG"
export PYTHONIOENCODING=utf-8

registra() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }

# Notebooks a executar: todos, ou só os números passados como argumento.
TODOS=(01_coleta_integracao 02_eda_preprocessamento 03_regressao 04_classificacao 05_previsao_2024)
if [ $# -gt 0 ]; then
  NOTEBOOKS=()
  for n in "$@"; do
    for nb in "${TODOS[@]}"; do
      [[ "$nb" == "$n"_* ]] && NOTEBOOKS+=("$nb")
    done
  done
  [ ${#NOTEBOOKS[@]} -eq 0 ] && { registra "nenhum notebook corresponde a: $* (use 01 a 05)"; exit 1; }
else
  NOTEBOOKS=("${TODOS[@]}")
fi

# Python 3.12: no Windows pelo launcher "py"; no Linux/macOS por python3.12 ou python3.
if command -v py > /dev/null 2>&1; then
  PY_BASE=(py -3.12)
elif command -v python3.12 > /dev/null 2>&1; then
  PY_BASE=(python3.12)
else
  PY_BASE=(python3)
fi

# Ambiente virtual em .venv (reaproveitado se já existir).
inicio=$(date +%s)
if [ ! -d .venv ]; then
  registra "criando o ambiente virtual em .venv"
  "${PY_BASE[@]}" -m venv .venv >> "$LOG" 2>&1 || { registra "FALHA ao criar o .venv"; exit 1; }
fi
if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi

registra "instalando as dependências (requirements.txt)"
"$PY" -m pip install -q --upgrade pip >> "$LOG" 2>&1
"$PY" -m pip install -q -r requirements.txt >> "$LOG" 2>&1 \
  || { registra "FALHA no pip install (veja executar.log; no Windows, use uma pasta de caminho curto)"; exit 1; }

cd notebooks || exit 1
for nb in "${NOTEBOOKS[@]}"; do
  t=$(date +%s)
  registra "executando $nb.ipynb"
  # --execute roda todas as células; --inplace grava as saídas no próprio notebook;
  # timeout de 1 h por célula, porque as buscas de hiperparâmetros passam de 10 min.
  if "../$PY" -m jupyter nbconvert --to notebook --execute --inplace \
       --ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=python3 \
       "$nb.ipynb" >> "$LOG" 2>&1; then
    registra "ok $nb ($(( $(date +%s) - t )) s)"
  else
    registra "FALHA em $nb (veja executar.log)"
    exit 1
  fi
done
registra "concluído: ${#NOTEBOOKS[@]} notebook(s) em $(( $(date +%s) - inicio )) s"
