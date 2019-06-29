class EggDrop:

    def __init__(self):
        self.eggCount = 2
        self.floor = -1
        self.gameEnd = false
        self.targetFloor = random.randint(1,101)

    def selectFloor(self,floor):
        if(floor > self.targetFloor):
            eggCOunt -= 1
            self.floor = floor

    def isGameEnd(self):
        if(eggCount == 0):
            return True

    def targetReached(self):
        if(self.floor == self.targetFloor):
            return True   

obj = EggDrop()
