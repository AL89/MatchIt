from pydantic import BaseModel

class Player(BaseModel):
    name: str
    win: int = 0
    draw: int = 0
    loss: int = 0

    def __eq__(self, other:"Player"):
        return self.name == other.name

    def __str__(self):
        return self.name

    def __repr__(self) -> str:
        repr_str = [str(self)]
        if self.win:
            repr_str.append(f"Wins: {self.win}")
        if self.loss:
            repr_str.append(f"Loss: {self.loss}")
        if self.draw:
            repr_str.append(f"Draw: {self.draw}")
        return "\n".join(repr_str)