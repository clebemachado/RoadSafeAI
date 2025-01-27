from abc import ABC, abstractmethod


class CollectData:
    
    @abstractmethod
    def execute(self, force_download=False):
        """Método que deve ser implementado pelas subclasses"""
        pass
        