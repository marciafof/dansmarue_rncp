from typing import List
import json

class ColumnType:
    def accept(self, _: str):
        return False
    def downgrade(self) -> "ColumnType":
        return self
    def sql(self) -> str:
        return ""

class Int(ColumnType):
    def accept(self, value: str):
        try:
            int(value)
            return True
        except ValueError:
            return False
    def downgrade(self) -> ColumnType:
        return Float()
    def sql(self) -> str:
        return "INT"

class Float(ColumnType):
    def accept(self, value: str):
        try:
            float(value)
            return True
        except ValueError:
            return False
    def downgrade(self) -> ColumnType:
        return Json()
    def sql(self) -> str:
        return "FLOAT"

class Json(ColumnType):
    def accept(self, value: str):
        try:
            json.loads(value)
            return True
        except json.decoder.JSONDecodeError:
            return False
    def downgrade(self) -> ColumnType:
        return Varchar()
    def sql(self) -> str:
        return "JSON"

class Varchar(ColumnType):
    length: int
    def __init__(self):
        self.length = 0
    def accept(self, value: str):
        self.length = max(self.length, len(value))
        return True
    def sql(self) -> str:
        return f"VARCHAR({self.length})"

class Predictor:
    def __init__(self):
        self.type: ColumnType = Int()
    def visit(self, value: str):
        while not self.type.accept(value):
            self.type = self.type.downgrade()

def predict(values: List[str]) -> ColumnType:
    pred = Predictor()
    for value in values:
        pred.visit(value)
    return pred.type
