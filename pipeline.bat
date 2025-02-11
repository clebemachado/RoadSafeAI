@echo off

REM Verifica se o ambiente virtual existe
if not exist env (
    echo Criando ambiente virtual...
    python -m venv env
	echo ativando ambiente virtual...
    call env\Scripts\activate
    call env\Scripts\python -m pip install --upgrade pip
	echo Instalando dependências...
    pip install -r requirements.txt
	echo Finalizado instalação de dependências...
) else (
    call env\Scripts\activate
    call env\Scripts\python -m pip install --upgrade pip
)

echo Iniciando processo da pipeline...
REM Executa o pipeline
python src\run_pipeline.py ^
    --random-state 42 ^
    --n-estimators 100 ^
    --output-dir "resultados" ^
    --test-size 0.2 ^
    --valid-size 0.2 ^
    --balance smote ^
    --dataset-type base ^
    --collect-new-data
	

call deactivate

echo Processo finalizado...