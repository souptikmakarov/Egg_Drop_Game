import random
import numpy as np
import pandas as pd
from keras.optimizers import Adam, RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import Activation
from keras import backend as K


# def AmaderRelu(x):
#     return K.relu(x, max_value=1.0, threshold=0.01)


class GameLearner:
    def __init__(self):
        self.reward = 0
        self.gamma = 0.8
        self.alpha = 0.8
        self.short_memory = np.array([])
        self.learning_rate = 0.0005
        self.model = self.network()
        # self.model = self.network("weights.hdf5")
        self.epsilon = 0
        self.actual = []
        self.memory = []

    def get_state(self, game):
        state = [game.eggCount, game.floor, game.progressing]
        return np.asarray(state)

    def set_reward(self, game):
        self.reward = 0
        extraPenalty = game.getRewardVal()
        # if game is won
        if game.targetReached():
            # if 1 egg remains
            if game.eggCount == 1:
                self.reward = 100
            # if 0 eggs remain
            elif game.eggCount == 0:
                self.reward = 50
        # if an egg was broken and it was a wrong guess
        elif game.isStateChange():
            # if the agent is guessing towards the correct answer
            if game.progressing:
                if game.eggCount == 1:
                    self.reward = -(5 + extraPenalty)
                elif game.eggCount == 0:
                    self.reward = -(15 + extraPenalty) #game end
            # if the agent is guessing away from the correct answer
            else:
                if game.eggCount == 1:
                    self.reward = -(20 + extraPenalty)
                elif game.eggCount == 0:
                    self.reward = -(35 + extraPenalty) #game end
        # if the egg did not break
        else:
            # if the agent is guessing towards the correct answer
            if game.progressing:
                if game.eggCount == 2:
                    self.reward = -5 + int(100/extraPenalty)
                elif game.eggCount == 1:
                    self.reward = -10 + int(100/extraPenalty)
            # if the agent is guessing away from the correct answer
            else:
                self.reward = - (5 + extraPenalty)
        return self.reward


    def network(self, weights=None):
        model = Sequential()
        model.add(Dense(output_dim=120, activation='relu', input_dim=3))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dropout(0.15))
        # output is 100 values, each value describing the probable reward on dropping the egg from that floor
        # model.add(Dense(output_dim=100, activation='softmax'))
        # opt = Adam(self.learning_rate)


        model.add(Dense(input_dim=120, output_dim=100, activation='relu'))
        opt = RMSprop(lr=self.learning_rate)

        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(weights)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory):
        if len(memory) > 1000:
            minibatch = random.sample(memory, 1000)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, 3)))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(state.reshape((1, 3)), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            pred = self.model.predict(next_state.reshape((1, 3)))[0]
            #alpha is the learning rate
            target = self.alpha * (reward + self.gamma * np.amax(pred))
        target_f = self.model.predict(state.reshape((1, 3)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, 3)), target_f, epochs=1, verbose=0)