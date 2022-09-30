from dataclasses import dataclass

@dataclass
class Params:
    name: str
    epochs: int
    batch_size: int
    learning_rate: float

