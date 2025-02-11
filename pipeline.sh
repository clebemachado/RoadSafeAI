#!/bin/bash

# Verifica se o ambiente virtual existe
if [ ! -d "env" ]; then
    echo "Criando ambiente virtual..."
    python -m venv env
    source env/Scripts/activate
    pip install -r requirements.txt
else
    source env/Scripts/activate
fi

# Executa o pipeline
python src\\run_pipeline.py \
    --random-state 42 \
    --n-estimators 100 \
    --output-dir "resultados" \
    --test-size 0.2 \
    --valid-size 0.2 \
    --balance smote \
    --dataset-type base \
    --collect-new-data

deactivate