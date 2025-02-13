from pathlib import Path

import pandas as pd

from config.config_project import ConfigProject
from config.inject_logger import inject_logger


@inject_logger
class FeatureEngineering:
    """
    Classe responsável pela criação de novas variáveis (feature engineering)
    para o dataset de acidentes.
    """
    
    def criar_periodo_dia(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria a variável periodo_dia a partir do horário.
        
        Períodos:
        - Madrugada: 00:00 - 05:59
        - Manhã: 06:00 - 11:59
        - Tarde: 12:00 - 17:59
        - Noite: 18:00 - 23:59
        """
        df = df.copy()
        
        def categorizar_periodo(horario):
            try:
                hora = int(horario.split(':')[0])
                if 0 <= hora < 6:
                    return 'madrugada'
                elif 6 <= hora < 12:
                    return 'manha'
                elif 12 <= hora < 18:
                    return 'tarde'
                else:
                    return 'noite'
            except:
                return None
        
        df['periodo_dia'] = df['horario'].apply(categorizar_periodo)
        
        # Log da distribuição dos períodos
        distribuicao = df['periodo_dia'].value_counts()
        self.logger.info("Distribuição dos períodos do dia:")
        df.drop(columns=['horario'], inplace=True)
        self.logger.info("Removendo horario exato, adotado periodo do dia:")
        for periodo, contagem in distribuicao.items():
            self.logger.info(f"- {periodo}: {contagem} registros")
            
        return df

    def criar_gravidade_acidente(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria a variável gravidade_acidente baseada no número de vítimas.
        
        Categorias:
        - Sem_vitimas: apenas ilesos
        - Leve: apenas feridos leves
        - Grave: presença de feridos graves
        - Fatal: presença de mortos
        
        Args:
            df: DataFrame com as colunas 'ilesos', 'feridos_leves', 'feridos_graves', 'mortos'
            
        Returns:
            DataFrame com a nova coluna 'gravidade_acidente'
        """
        df = df.copy()
        
        # Garantir que valores nulos sejam tratados como zero
        colunas_vitimas = ['ilesos', 'feridos_leves', 'feridos_graves', 'mortos']
        for col in colunas_vitimas:
            df[col] = df[col].fillna(0).astype(int)
        
        # Criar a classificação de gravidade
        def classificar_gravidade(row):
            if row['mortos'] > 0:
                return 'fatal'
            elif row['feridos_graves'] > 1 and row['feridos_leves']>= 3:
                return 'grave'
            elif row['feridos_leves'] > 0 and row['feridos_graves'] == 1:
                return 'moderado'
            else:
                return 'leve'
        
        df['gravidade_acidente'] = df.apply(classificar_gravidade, axis=1)
        
        # Log da distribuição das gravidades
        distribuicao = df['gravidade_acidente'].value_counts()
        self.logger.info("Distribuição das gravidades dos acidentes:")
        for gravidade, contagem in distribuicao.items():
            self.logger.info(f"- {gravidade}: {contagem} registros")
        
        return df
    
    def tratar_causas_acidente(self, df: pd.DataFrame, min_frequency: int = 10) -> pd.DataFrame:
        """
        Trata a coluna causa_acidente agrupando causas similares e tratando valores raros.
        
        Args:
            df: DataFrame com a coluna 'causa_acidente'
            min_frequency: Frequência mínima para manter uma categoria
            
        Returns:
            DataFrame com a coluna 'causa_acidente_grupo' adicionada
        """
        df = df.copy()
        
        # Mapeamento de categorias similares
        CAUSA_MAPPING = {
            # Falhas Humanas - Atenção
            'Falta de atenção': 'falha_atencao',
            'Falta de Atenção à Condução': 'falha_atencao',
            'Reação tardia ou ineficiente do condutor': 'falha_atencao',
            'Ausência de reação do condutor': 'falha_atencao',
            
            # Falhas Humanas - Comportamento de Risco
            'Não guardar distância de segurança': 'comportamento_risco',
            'Ultrapassagem indevida': 'comportamento_risco',
            'Velocidade incompatível': 'comportamento_risco',
            'Desobediência à sinalização': 'comportamento_risco',
            'Desobediência às normas de trânsito pelo condutor': 'comportamento_risco',
            'Participar de racha': 'comportamento_risco',
            
            # Fatores Externos - Animais
            'Animais na Pista': 'fatores_externos_animais',
            
            # Problemas Técnicos - Veículo
            'Defeito mecânico em veículo': 'problemas_tecnicos_veiculo',
            'Problema na suspensão': 'problemas_tecnicos_veiculo',
            'Avarias e/ou desgaste excessivo no pneu': 'problemas_tecnicos_veiculo',
            'Problema com o freio': 'problemas_tecnicos_veiculo',
            
            # Problemas Técnicos - Via
            'Defeito na via': 'problemas_tecnicos_via',
            'Defeito na Via': 'problemas_tecnicos_via',
            'Pista Escorregadia': 'problemas_tecnicos_via',
            'Acumulo de água sobre o pavimento': 'problemas_tecnicos_via',
            'Acumulo de óleo sobre o pavimento': 'problemas_tecnicos_via',
            
            # Condições do Condutor - Álcool e Drogas
            'Ingestão de álcool': 'condutor_alcool_drogas',
            'Ingestão de substâncias psicoativas': 'condutor_alcool_drogas',
            'Ingestão de Álcool': 'condutor_alcool_drogas',
            'Ingestão de Substâncias Psicoativas': 'condutor_alcool_drogas',
            
            # Condições do Condutor - Fadiga
            'Dormindo': 'condutor_fadiga',
            'Condutor Dormindo': 'condutor_fadiga',
            
            # Fatores Ambientais
            'Neblina': 'fatores_ambientais',
            'Chuva': 'fatores_ambientais',
            'Fumaça': 'fatores_ambientais',
            
            # Outros
            'Outras': 'outros'
        }
        
        # Criar nova coluna com o agrupamento
        df['causa_acidente_grupo'] = df['causa_acidente'].map(CAUSA_MAPPING)
        
        # Tratar valores que não foram mapeados
        causas_freq = df['causa_acidente'].value_counts()
        causas_raras = causas_freq[causas_freq < min_frequency].index
        
        # Mapear causas raras para 'outros'
        df.loc[df['causa_acidente'].isin(causas_raras), 'causa_acidente_grupo'] = 'outros'
        
        # Preencher valores não mapeados com 'outros'
        df['causa_acidente_grupo'] = df['causa_acidente_grupo'].fillna('outros')
        
        # Log da distribuição final
        distribuicao = df['causa_acidente_grupo'].value_counts()
        self.logger.info("Distribuição final das causas agrupadas:")
        for causa, contagem in distribuicao.items():
            self.logger.info(f"- {causa}: {contagem} registros")
            
        return df
    
    def salvar_dataset(self, df: pd.DataFrame, nome_arquivo: str) -> None:
        """
        Salva o DataFrame processado em um arquivo CSV usando o caminho definido no config.yaml.
        """
        config = ConfigProject()
        
        pasta_destino = config.get("paths.output_files")
        
        if not pasta_destino:
            self.logger.warning("Caminho de output não encontrado no config.yaml. Usando caminho padrão 'files/processed'")
            pasta_destino = "files/processed"
        
        notebook_path = Path().absolute()  # Caminho atual
        project_root = notebook_path.parent.parent  # Sobe dois níveis (notebooks/main -> root)
        output_path = project_root / pasta_destino.strip("./")  # Remove ./ se existir e concatena
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        caminho_arquivo = output_path / f"{nome_arquivo}.csv"
        
        df.to_csv(caminho_arquivo, index=False)
        self.logger.info(f"Dataset processado salvo em: {caminho_arquivo}")

    def criar_todas_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica todas as transformações de feature engineering no dataset.
        """
        self.logger.info("Iniciando criação de novas features...")
        
        # Criar período do dia
        df = self.criar_periodo_dia(df)
        self.logger.info("Feature 'periodo_dia' criada com sucesso")
        
        # Criar gravidade do acidente
        df = self.criar_gravidade_acidente(df)
        self.logger.info("Feature 'gravidade_acidente' criada com sucesso")
        
        df = self.tratar_causas_acidente(df)
        self.logger.info("Feature 'causa_acidente_grupo' criada com sucesso")


        # Validação final
        novas_colunas = ['periodo_dia', 'gravidade_acidente','causa_acidente_grupo']
        colunas_ausentes = [col for col in novas_colunas if col not in df.columns]
        
        if colunas_ausentes:
            self.logger.warning(f"Atenção: As seguintes colunas não foram criadas: {colunas_ausentes}")
        else:
            self.logger.info("Todas as features foram criadas com sucesso!")
        
        self.salvar_dataset(df, 'datatran_ma_processado')
        return df
