from random import randint
from GameLearner import GameLearner
import numpy as np
from GameRecordDataModel import GameRecordData, Prediction
import json
from datetime import datetime

class EggDrop:

    def __init__(self):
        self.eggCount = 2
        self.floor = -1
        self.prevFloor = 0
        self.gameEnd = False
        self.stateChange = False
        self.progressing = True
        # self.targetFloor = randint(1, 100)
        self.targetFloor = 35

    def selectFloor(self, floor):
        self.prevFloor = self.floor
        self.floor = floor

        #if egg is broken
        if floor >= self.targetFloor:
            self.eggCount -= 1
            self.stateChange = True
            # if last prediction was a higher floor than the current prediction
            if self.prevFloor > self.floor:
                self.progressing = True
            # if last prediction was a lower floor than the current prediction or the same value
            else:
                self.progressing = False
        # if egg is not broken
        else:
            self.stateChange = False
            # if last prediction was a higher floor than the current prediction or the same value
            if self.prevFloor >= self.floor:
                self.progressing = False
            # if last prediction was a lower floor than the current prediction
            else:
                self.progressing = True

    def isStateChange(self):
        retVal = self.stateChange
        self.stateChange = False
        return retVal

    def isGameEnd(self):
        return self.eggCount == 0

    def targetReached(self):
        return self.floor == self.targetFloor

    def getRewardVal(self):
        return np.abs(self.targetFloor - self.floor)


def printGameData(data):
    predStr = ""
    for pred in data.Predictions:
        predStr += "{},{},{}".format(pred.Floor, pred.EggsLeft, pred.Reward) + "||"

    print("{}~~~{}~~~{}".format(data.GameId, data.ActualAnswer, predStr))
    with open('GameRunLog.txt', 'a') as f:
        f.write("{}~~~{}~~~{}\n".format(data.GameId, data.ActualAnswer, predStr))


def run():
    agent = GameLearner()
    # print(agent.model.summary())
    counter_games = 0
    gameRunData = []
    while counter_games < 2000:
        #Init Game Classes
        game = EggDrop()
        gameData = GameRecordData()
        gameData.GameId = counter_games
        gameData.ActualAnswer = game.targetFloor
        gameMoveId = 1
        while not game.isGameEnd():
            # agent.epsilon is set to give randomness to actions
            agent.epsilon = 80 - counter_games

            # get old state
            state_old = agent.get_state(game)

            # perform random actions based on agent.epsilon, or choose the action
            if randint(0, 200) < agent.epsilon:
                final_move = randint(0, 99)
            else:
                # predict action based on the old state
                prediction = agent.model.predict(state_old.reshape((1, 3)))
                # take the floor value for which predicted reward is highest
                final_move = np.argmax(prediction[0])

            # perform new move and get new state
            game.selectFloor(final_move + 1)
            state_new = agent.get_state(game)

            # set reward for the new state
            reward = agent.set_reward(game)

            moveData = Prediction()
            moveData.EggsLeft = game.eggCount
            moveData.Floor = game.floor
            moveData.Id = gameMoveId
            moveData.Reward = reward
            gameMoveId += 1
            gameData.Predictions.append(moveData)

            # train short memory base on the new action and state
            crash = game.isGameEnd() or game.targetReached()
            agent.train_short_memory(state_old, final_move, reward, state_new, crash)

            # store the new data into a long term memory
            agent.remember(state_old, final_move, reward, state_new, crash)
            if crash or gameMoveId > 100:
                break

        agent.replay_new(agent.memory)
        counter_games += 1
        gameRunData.append(gameData)
        printGameData(gameData)

    with("GameRunLog-{}.json".format(datetime.now().strftime("%d-%m-%Y|%H-%M-%S")), 'w') as f:
        json.dump(gameRunData, f)


run()
