
class TrainerConfiguration:
    name: str
    configuration: dict
    optimizer = None

    def __init__(self, name: str, configuration: dict):
        self.name = name
        self.configuration = configuration
