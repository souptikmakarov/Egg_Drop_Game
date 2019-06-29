class GameRecordData:
    def __init__(self):
        self.GameId = -1
        self.Predictions = []
        self.ActualAnswer = -1

class Prediction:
    def __init__(self):
        self.Id = -1
        self.Floor = -1
        self.EggsLeft = -1
        self.Reward = -1