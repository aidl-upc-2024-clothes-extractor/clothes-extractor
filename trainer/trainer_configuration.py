from abc import ABC
import string


class TrainerConfiguration(ABC):
    name: string
    configuration: dict

    def __init__(self, name: string, configuration: dict):
        self.name = name
        self.configuration = configuration
