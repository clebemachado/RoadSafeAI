import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NivelGravidade(Enum):
    """Enum para definir os níveis de gravidade dos acidentes"""
    BAIXA = "Baixa"
    MEDIA = "Média"
    ALTA = "Alta"
    GRAVISSIMA = "Gravíssima"

@dataclass
class ConfiguracoesGravidade:
    """Classe para armazenar as configurações de classificação de gravidade"""
    limiar_mortos: int = 1
    limiar_feridos_graves: int = 1
    limiar_feridos_leves: int = 1

class DadosInvalidosError(Exception):
    """Exceção personalizada para dados inválidos"""
    pass

class ProcessadorAcidentesGravidade:
    """Classe principal para processamento dos dados de acidentes"""
    
    def __init__(self, config: Optional[ConfiguracoesGravidade] = None):
        """
        Inicializa o processador com configurações personalizadas ou padrão
        
        Args:
            config: Configurações para classificação de gravidade
        """
        self.config = config or ConfiguracoesGravidade()
        self.logger = logging.getLogger(__name__)
        
    def carregar_dados(self, caminho_arquivo: str) -> pd.DataFrame:
        """
        Carrega os dados do arquivo CSV
        
        Args:
            caminho_arquivo: Caminho para o arquivo CSV
            
        Returns:
            DataFrame com os dados carregados
            
        Raises:
            FileNotFoundError: Se o arquivo não for encontrado
            DadosInvalidosError: Se os dados estiverem em formato inválido
        """
        try:
            df = pd.read_csv(caminho_arquivo, sep=";")
            if df.empty:
                raise DadosInvalidosError("O arquivo CSV está vazio")
                
            self.logger.info(f"Dados carregados com sucesso. Total de registros: {len(df)}")
            return df
            
        except pd.errors.EmptyDataError:
            self.logger.error("O arquivo CSV está vazio")
            raise DadosInvalidosError("O arquivo CSV está vazio")
        except FileNotFoundError:
            self.logger.error(f"Arquivo não encontrado: {caminho_arquivo}")
            raise
        except Exception as e:
            self.logger.error(f"Erro ao carregar arquivo: {str(e)}")
            raise
    
    def _validar_dataframe(self, df: Optional[pd.DataFrame]) -> None:
        """
        Valida se o DataFrame é válido para processamento
        
        Args:
            df: DataFrame a ser validado
            
        Raises:
            DadosInvalidosError: Se o DataFrame for inválido
        """
        if df is None:
            raise DadosInvalidosError("DataFrame não pode ser None")
        if not isinstance(df, pd.DataFrame):
            raise DadosInvalidosError("O objeto não é um DataFrame válido")
        if df.empty:
            raise DadosInvalidosError("DataFrame está vazio")
    
    def _calcular_gravidade(self, row: pd.Series) -> str:
        """
        Calcula o nível de gravidade para uma linha específica
        
        Args:
            row: Série do pandas com os dados de um acidente
            
        Returns:
            String representando o nível de gravidade
        """
        try:
            mortos = pd.to_numeric(row['mortos'], errors='coerce') or 0
            feridos_graves = pd.to_numeric(row['feridos_graves'], errors='coerce') or 0
            feridos_leves = pd.to_numeric(row['feridos_leves'], errors='coerce') or 0
            classificacao = str(row['classificacao_acidente'])
            
            if mortos >= self.config.limiar_mortos or 'Fatais' in classificacao:
                return NivelGravidade.GRAVISSIMA.value
            elif feridos_graves >= self.config.limiar_feridos_graves:
                return NivelGravidade.ALTA.value
            elif feridos_leves >= self.config.limiar_feridos_leves:
                return NivelGravidade.MEDIA.value
            else:
                return NivelGravidade.BAIXA.value
        except Exception as e:
            self.logger.warning(f"Erro ao processar valores para linha: {str(e)}")
            return NivelGravidade.BAIXA.value
    
    def adicionar_gravidade(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona a coluna de gravidade ao DataFrame
        
        Args:
            df: DataFrame com os dados originais
            
        Returns:
            DataFrame com a nova coluna de gravidade
            
        Raises:
            DadosInvalidosError: Se o DataFrame for inválido
        """
        try:
            # Valida o DataFrame antes do processamento
            self._validar_dataframe(df)
            
            self.logger.warning(f"Colunas dataframe: {str(df.columns)}")

            # Verifica se as colunas necessárias existem
            colunas_necessarias = ['mortos','feridos_leves','feridos_graves','classificacao_acidente']
            colunas_faltantes = [col for col in colunas_necessarias if col not in df.columns]
            
            if colunas_faltantes:
                raise DadosInvalidosError(f"Colunas faltantes no DataFrame: {colunas_faltantes}")
            
            # Faz uma cópia do DataFrame para evitar modificações no original
            df_processado = df.copy()
            
            # Adiciona a coluna de gravidade
            df_processado['gravidade'] = df_processado.apply(self._calcular_gravidade, axis=1)
            
            self.logger.info("Coluna de gravidade adicionada com sucesso")
            return df_processado
            
        except Exception as e:
            self.logger.error(f"Erro ao adicionar coluna de gravidade: {str(e)}")
            raise
    
    def gerar_estatisticas(self, df: pd.DataFrame) -> Dict:
        """
        Gera estatísticas sobre os níveis de gravidade
        
        Args:
            df: DataFrame com a coluna de gravidade
            
        Returns:
            Dicionário com estatísticas sobre a distribuição de gravidade
            
        Raises:
            DadosInvalidosError: Se o DataFrame for inválido ou não contiver a coluna de gravidade
        """
        try:
            self._validar_dataframe(df)
            
            if 'gravidade' not in df.columns:
                raise DadosInvalidosError("Coluna 'gravidade' não encontrada no DataFrame")
            
            stats = {
                'distribuicao_gravidade': df['gravidade'].value_counts().to_dict(),
                'percentual_gravidade': df['gravidade'].value_counts(normalize=True).to_dict(),
                'total_acidentes': len(df)
            }
            
            self.logger.info("Estatísticas geradas com sucesso")
            return stats
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar estatísticas: {str(e)}")
            raise

def main():
    """Função principal para demonstração do uso"""
    processador = ProcessadorAcidentesGravidade()
    
    try:
        # Carrega os dados
        df = processador.carregar_dados('datatran2024.csv')
        
        # Adiciona a coluna de gravidade
        df_processado = processador.adicionar_gravidade(df)
        
        # Gera estatísticas
        estatisticas = processador.gerar_estatisticas(df_processado)
        
        # Salva o resultado
        df_processado.to_csv('acidentes_com_gravidade.csv', index=False)
        
        # Exibe algumas estatísticas
        print("\nEstatísticas de Gravidade:")
        for nivel, quantidade in estatisticas['distribuicao_gravidade'].items():
            percentual = estatisticas['percentual_gravidade'][nivel] * 100
            print(f"{nivel}: {quantidade} acidentes ({percentual:.1f}%)")
            
    except DadosInvalidosError as e:
        logging.error(f"Erro de dados inválidos: {str(e)}")
    except Exception as e:
        logging.error(f"Erro durante o processamento: {str(e)}")
        raise

if __name__ == "__main__":
    main()