from abc import ABC


class TrainerConfiguration(ABC):
    name: str
    configuration: dict

    def __init__(self, name: str, configuration: dict):
        self.name = name
        self.configuration = configuration
