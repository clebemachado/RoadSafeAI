from abc import ABC, abstractmethod


class CollectData:
    
    @abstractmethod
    def execute(self):
        """Método que deve ser implementado pelas subclasses"""
        pass
        