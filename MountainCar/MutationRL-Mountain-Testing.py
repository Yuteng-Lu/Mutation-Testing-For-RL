#  A car is on a one-dimensional track, positioned between two "mountains".
#  The goal is to drive up the mountain on the right; however,
#  the car's engine is not strong enough to scale the mountain in a single pass. Therefore,
#  the only way to succeed is to drive back and forth to build up momentum.


import gym
import random
from keras import Sequential
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.activations import relu, linear

import numpy as np
env = gym.make('MountainCar-v0')
env.seed(110)
np.random.seed(10)
TURN_LIMIT = 200
TEST_ROUND = 100
# threshold for testing trained agent
threshold = -180


class DQN:

    """ Implementation of deep q learning algorithm """

    def __init__(self, action_space, state_space):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1.0
        self.gamma = .95
        self.batch_size = 64
        self.epsilon_min = .01
        self.lr = 0.001
        self.epsilon_decay = .995
        self.memory = deque(maxlen=100000)
        self.model = self.build_model()

    def build_model(self):

        model = Sequential()
        model.add(Dense(20, input_dim=self.state_space, activation=relu))
        model.add(Dense(25, activation=relu))
        model.add(Dense(self.action_space, activation=linear))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
        
    def act_test(self, state):

        act_values = self.model.predict(state)
        return np.argmax(act_values[0]) 

    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states) # [[1 2], [3 4]]
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)
        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def reward_abnormal_replay(self):

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states) # [[1 2], [3 4]]
        next_states = np.squeeze(next_states)

        targets = -1 * rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)
        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def get_reward(state):

    if state[0] >= 0.5:
        # "Car has reached the goal"
        return 100
    if state[0] > -0.4:
        return (1+state[0])**2
    return 0

def get_reward_for_TE(state):
# Changing the reward rules will affect the agent's cognition of the environment.
    if state[0] >= 0.3:
        # "Car has reached the goal"
        return 100
    if state[0] > -0.4:
        return (1+state[0])**2
    return 0

def train_and_test_RA(episode):
# reward abnormal
    train_scores = []
    agent = DQN(env.action_space.n, env.observation_space.shape[0])
    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, 2))
        score = 0
        max_steps = 201
        for i in range(max_steps):
            action = agent.act(state) # return 0, 1, 2
            # env.render()
            next_state, reward, done, _ = env.step(action)
            reward = get_reward(next_state) # next_state = [-5.27085603e-01  2.63483272e-05]
            score += reward
            next_state = np.reshape(next_state, (1, 2))
            if i < max_steps * 0.75:
                agent.remember(state, action, reward, next_state, done)
            else:
                agent.remember(state, action, 0, next_state, done)
            state = next_state
            agent.replay()
            if done:
                # print("episode: {}/{}, score: {}".format(e, episode, score))
                break
    record = 0
    for i in range(TEST_ROUND):
        test_score = 0
        state = env.reset()
        for t in range(TURN_LIMIT):
            state = np.reshape(state, (1, 2))
            act = agent.act_test(state)
            next_state, reward, done, info = env.step(act)
            test_score += reward
            if done and test_score >= threshold:
                record += 1
                break
            else:
                state = next_state
    return record

def train_and_test(episode):

    train_scores = []
    agent = DQN(env.action_space.n, env.observation_space.shape[0])
    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, 2))
        score = 0
        max_steps = 201
        for i in range(max_steps):
            action = agent.act(state) # return 0, 1, 2
            # env.render()
            next_state, reward, done, _ = env.step(action)
            reward = get_reward(next_state) 
            score += reward
            next_state = np.reshape(next_state, (1, 2))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            if done:
                # print("episode: {}/{}, score: {}".format(e, episode, score))
                break
    record = 0
    for i in range(TEST_ROUND):
        test_score = 0
        state = env.reset()
        for t in range(TURN_LIMIT):
            state = np.reshape(state, (1, 2))
            act = agent.act_test(state)
            next_state, reward, done, info = env.step(act)
            test_score += reward
            if done and test_score >= threshold:
                record += 1
                break
            else:
                state = next_state
    return record

def train_and_test_RA_In_TestEnv(episode):
# reward abnormal
    train_scores = []
    agent = DQN(env.action_space.n, env.observation_space.shape[0])
    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, 2))
        score = 0
        max_steps = 201
        for i in range(max_steps):
            action = agent.act(state) # return 0, 1, 2
            # env.render()
            next_state, reward, done, _ = env.step(action)
            reward = get_reward_for_TE(next_state) 
            score += reward
            next_state = np.reshape(next_state, (1, 2))
            if i < max_steps * 0.75:
                agent.remember(state, action, reward, next_state, done)
            else:
                agent.remember(state, action, 0, next_state, done)
            state = next_state
            agent.replay()
            if done:
                # print("episode: {}/{}, score: {}".format(e, episode, score))
                break
    record = 0
    for i in range(TEST_ROUND):
        test_score = 0
        state = env.reset()
        for t in range(TURN_LIMIT):
            state = np.reshape(state, (1, 2))
            act = agent.act_test(state)
            next_state, reward, done, info = env.step(act)
            test_score += reward
            if done and test_score >= threshold:
                record += 1
                break
            else:
                state = next_state
    return record

def train_and_test_In_TestEnv(episode):

    train_scores = []
    agent = DQN(env.action_space.n, env.observation_space.shape[0])
    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, 2))
        score = 0
        max_steps = 201
        for i in range(max_steps):
            action = agent.act(state) # return 0, 1, 2
            # env.render()
            next_state, reward, done, _ = env.step(action)
            reward = get_reward_for_TE(next_state) 
            score += reward
            next_state = np.reshape(next_state, (1, 2))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            if done:
                # print("episode: {}/{}, score: {}".format(e, episode, score))
                break
    record = 0
    for i in range(TEST_ROUND):
        test_score = 0
        state = env.reset()
        for t in range(TURN_LIMIT):
            state = np.reshape(state, (1, 2))
            act = agent.act_test(state)
            next_state, reward, done, info = env.step(act)
            test_score += reward
            if done and test_score >= threshold:
                record += 1
                break
            else:
                state = next_state
    return record

def random_policy(episode, step):
    i = 0
    for i_episode in range(episode):
        env.reset()
        for t in range(step):
            # env.render()
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            if done:
                i += 1
                # print("Episode finished after {} timesteps".format(t+1))
                break
    return i
           
if __name__ == '__main__':

    episodes, Record = 100, 0
    Agent_amount = 20
    # agent = DQN(env.action_space.n, env.observation_space.shape[0])

    for i in range(Agent_amount):
        # Run the code below to see what happens when the system is normal.
        # record = train_and_test(episodes)

        # If you need to switch to mutation mode, run the code below.
        # record = train_and_test_RA(episodes)

        # How a normal system behaves in the test environment
        # record = train_and_test_In_TestEnv(episodes)

        # How a mutated system behaves in the test environment
        record = train_and_test_RA_In_TestEnv(episodes)

        Record += record

    print(Record / Agent_amount) # (Record / Agent_amount) measures the average performance of Agent_amount agents
    