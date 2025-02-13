import pandas as pd
from config.inject_logger import inject_logger
@inject_logger
class DataStandardize:
    """
    Classe responsável pela padronização dos dados do dataset de acidentes.
    """
    
    # Colunas numéricas que precisam ser verificadas/ajustadas
    NUMERIC_COLUMNS = [
        'km', 'pessoas', 'mortos', 'feridos_leves', 
        'feridos_graves', 'ilesos', 'ignorados', 'feridos', 'veiculos'
    ]
    
    # Mapeamento para uso_solo
    USO_SOLO_MAPPING = {
        'Rural': 'Não',
        'Urbano': 'Sim',
        'Não': 'Não',
        'Sim': 'Sim'
    }
    
    # Mapeamento para dias da semana
    DIAS_SEMANA_MAPPING = {
        'Segunda': 'segunda-feira',
        'Terça': 'terca-feira',
        'Quarta': 'quarta-feira',
        'Quinta': 'quinta-feira',
        'Sexta': 'sexta-feira',
        'Sábado': 'sabado',
        'Domingo': 'domingo',
        'domingo': 'domingo',
        'segunda-feira': 'segunda-feira',
        'terça-feira': 'terca-feira',
        'quarta-feira': 'quarta-feira',
        'quinta-feira': 'quinta-feira',
        'sexta-feira': 'sexta-feira',
        'sábado': 'sabado'
    }

    def padronizar_valores_numericos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Padroniza os valores numéricos do dataset, convertendo para o tipo correto
        e tratando possíveis inconsistências.
        
        Args:
            df: DataFrame com as colunas numéricas
            
        Returns:
            DataFrame com valores numéricos padronizados
        """
        df = df.copy()
        
        for col in DataStandardize.NUMERIC_COLUMNS:
            if col in df.columns:
                # Converter para string primeiro para garantir consistência no tratamento
                df[col] = df[col].astype(str)
                
                # Remover caracteres não numéricos e espaços
                df[col] = df[col].str.replace(',', '.')
                df[col] = df[col].str.extract('(\d+\.?\d*)', expand=False)
                
                # Converter para float e depois para int
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Todas as colunas numéricas serão inteiras
                df[col] = df[col].fillna(0).astype(int)
                
                self.logger.info(f"Coluna {col} padronizada. Range: [{df[col].min()}, {df[col].max()}]")
        
        return df

    def padronizar_valores_temporais(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Padroniza as colunas temporais do dataset.
        - Converte data_inversa para datetime
        - Cria coluna periodo_dia baseada no horário
        
        Args:
            df: DataFrame com as colunas temporais
            
        Returns:
            DataFrame com valores temporais padronizados
        """
        df = df.copy()
        
        # Converter data_inversa para datetime
        if 'data_inversa' in df.columns:
            df['data'] = pd.to_datetime(df['data_inversa'])
            df.drop('data_inversa', axis=1, inplace=True)
            self.logger.info("Coluna data_inversa convertida para datetime e renomeada para 'data'")
        
        # Criar período do dia
        if 'horario' in df.columns:
            def get_periodo(horario):
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
            
            df['periodo_dia'] = df['horario'].apply(get_periodo)
            
            # Log da distribuição dos períodos
            periodo_counts = df['periodo_dia'].value_counts()
            self.logger.info("Distribuição dos períodos do dia:")
            for periodo, count in periodo_counts.items():
                self.logger.info(f"- {periodo}: {count}")
        
        return df

    def padronizar_uso_solo(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Padroniza a coluna uso_solo conforme dicionário da PRF.
        Rural -> Não
        Urbano -> Sim
        
        Args:
            df: DataFrame com a coluna uso_solo
            
        Returns:
            DataFrame com uso_solo padronizado
        """
        df = df.copy()
        
        if 'uso_solo' in df.columns:
            # Registrar valores únicos antes da transformação
            valores_originais = df['uso_solo'].unique()
            self.logger.info(f"Valores únicos originais em uso_solo: {valores_originais}")
            
            # Aplicar mapeamento
            df['uso_solo'] = df['uso_solo'].replace(DataStandardize.USO_SOLO_MAPPING)
            
            # Registrar valores únicos após a transformação
            valores_finais = df['uso_solo'].unique()
            self.logger.info(f"Valores únicos após padronização em uso_solo: {valores_finais}")
            
            # Verificar se existem valores não mapeados
            valores_nao_mapeados = set(valores_originais) - set(DataStandardize.USO_SOLO_MAPPING.keys())
            if valores_nao_mapeados:
                self.logger.warning(f"Valores não mapeados em uso_solo: {valores_nao_mapeados}")
        
        return df

    def padronizar_dia_semana(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Padroniza os valores da coluna dia_semana para um formato único.
        
        Args:
            df: DataFrame com a coluna dia_semana
            
        Returns:
            DataFrame com dia_semana padronizado
        """
        df = df.copy()
        
        if 'dia_semana' in df.columns:
            # Registrar valores únicos antes da transformação
            valores_originais = df['dia_semana'].unique()
            self.logger.info(f"Valores únicos originais em dia_semana: {valores_originais}")
            
            # Aplicar mapeamento
            df['dia_semana'] = df['dia_semana'].replace(DataStandardize.DIAS_SEMANA_MAPPING)
            
            # Registrar valores únicos após a transformação
            valores_finais = df['dia_semana'].unique()
            self.logger.info(f"Valores únicos após padronização em dia_semana: {valores_finais}")
            
            # Verificar se existem valores não mapeados
            valores_nao_mapeados = set(valores_originais) - set(DataStandardize.DIAS_SEMANA_MAPPING.keys())
            if valores_nao_mapeados:
                self.logger.warning(f"Valores não mapeados em dia_semana: {valores_nao_mapeados}")
        
        return df

    def padronizar_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica todas as padronizações no dataset.
        
        Args:
            df: DataFrame original
            
        Returns:
            DataFrame padronizado
        """
        self.logger.info("Iniciando padronização do dataset...")
        
        # Padronizar valores numéricos
        df = self.padronizar_valores_numericos(df)
        self.logger.info("Valores numéricos padronizados")
        
        # Padronizar valores temporais
        df = self.padronizar_valores_temporais(df)
        self.logger.info("Valores temporais padronizados")
        
        # Padronizar uso_solo
        df = self.padronizar_uso_solo(df)
        self.logger.info("Coluna uso_solo padronizada")
        
        # Padronizar dia_semana
        df = self.padronizar_dia_semana(df)
        self.logger.info("Coluna dia_semana padronizada")
        
        return df