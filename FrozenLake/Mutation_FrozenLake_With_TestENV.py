# The parameters and results of this version are meaningful
import sys
from contextlib import closing

import gym
import random
import numpy as np
from io import StringIO

from gym import utils
from gym.envs.toy_text import discrete
from gym import wrappers
from random import choice

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {

    "4x4": [
        "SHFG",
        "FHFF",
        "FFFH",
        "HFFH"
    ],
}


def generate_random_map(size=8, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # DFS to check that it's a valid path.
    def is_valid(res):
        frontier, discovered = [], set()
        frontier.append((0, 0))
        while frontier:
            r, c = frontier.pop()
            # row, column
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == 'G':
                        return True
                    if (res[r_new][c_new] != 'H'):
                        frontier.append((r_new, c_new))
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(['F', 'H'], (size, size), p=[p, 1-p])
        res[0][0] = 'S'
        res[-1][-1] = 'G'
        valid = is_valid(res)
    return ["".join(x) for x in res]


class FrozenLakeEnv(discrete.DiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the
    park when you made a wild throw that left the frisbee out in the middle of
    the lake. The water is mostly frozen, but there are a few holes where the
    ice has melted. If you step into one of those holes, you'll fall into the
    freezing water. At this time, there's an international frisbee shortage, so
    it's absolutely imperative that you navigate across the lake and retrieve
    the disc. However, the ice is slippery, so you won't always move in the
    direction you intend.
    The surface is described using a grid like the following
        SFFF
        FHFH
        FFFH
        HFFG
    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located
    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4", is_slippery=True):
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]
            done = bytes(newletter) in b'GH'
            reward = float(newletter == b'G')
            return newstate, reward, done

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'GH':
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                li.append((
                                    1. / 3.,
                                    *update_probability_matrix(row, col, b)
                                ))
                        else:
                            li.append((
                                1., *update_probability_matrix(row, col, a)
                            ))

        super(FrozenLakeEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(
                ["Left", "Down", "Right", "Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()

# Q learning params
ALPHA = 0.1 # learning rate
GAMMA = 0.99 # reward discount
LEARNING_COUNT = 10000
TEST_COUNT = 100
AGENT_AMOUNT = 11

TURN_LIMIT = 500
IS_MONITOR = True

class Agent:
    def __init__(self, env):
        self.env = env
        self.episode_reward = 0.0
        self.q_val = np.zeros(16 * 4).reshape(16, 4).astype(np.float32)

    def learn(self):
        # one episode learning
        state = self.env.reset()
        # self.env.render()   
        for t in range(TURN_LIMIT):
            act = self.env.action_space.sample() # random
            next_state, reward, done, info = self.env.step(act)  
            # Q <- Q + a(Q' - Q)
            # <=> Q <- (1-a)Q + a(Q')
            # Normal condition
            q_next_max = np.max(self.q_val[next_state])  
            self.q_val[state][act] = (1 - ALPHA) * self.q_val[state][act]\
                                 + ALPHA * (reward + GAMMA * q_next_max) 
            # self.env.render()
            if done:
                return reward
            else: 
                state = next_state

    def learn_state_error(self):
        # one episode learning
        state = self.env.reset()
        # self.env.render()
        # Observation(State) error 
        container_state = []
        for t in range(TURN_LIMIT):
            # Observation(State) error 
            container_state.append(state)
            act = self.env.action_space.sample() # random
            next_state, reward, done, info = self.env.step(act)
            # Observation(State) error 
            if t < 12:
                error_next_state = choice(container_state)
                error_q_next_max = np.max(self.q_val[error_next_state])           
                self.q_val[state][act] = (1 - ALPHA) * self.q_val[state][act]\
                                        + ALPHA * (reward + GAMMA * error_q_next_max)   
            else:
                q_next_max = np.max(self.q_val[next_state])   
                self.q_val[state][act] = (1 - ALPHA) * self.q_val[state][act]\
                                        + ALPHA * (reward + GAMMA * error_q_next_max) 
            # self.env.render()
            if done:
                return reward
            else: 
                state = next_state

    def learn_state_repetition(self):
        # one episode learning
        state = self.env.reset()
        # self.env.render()
        for t in range(TURN_LIMIT):
            act = self.env.action_space.sample() # random
            next_state, reward, done, info = self.env.step(act)
            # Observation(State) repetition  
            if t > 5: 
                q_next_max = np.max(self.q_val[state])
            else:
                q_next_max = np.max(self.q_val[next_state])
            # self.env.render()
            if done:
                return reward
            else: 
                state = next_state

    def learn_state_crash(self):
        # one episode learning
        state = self.env.reset()
        # self.env.render()

        for t in range(TURN_LIMIT):
            act = self.env.action_space.sample() # random
            next_state, reward, done, info = self.env.step(act)

            # Observation(State) loss
            if t > 12: 
                q_next_max = np.max(self.q_val[next_state])  
                self.q_val[state][act] = (1 - ALPHA) * self.q_val[state][act]\
                                    + ALPHA * (reward + GAMMA * q_next_max)       
            # self.env.render()
            if done:
                return reward
            else: 
                state = next_state

    def learn_state_delay(self):
        # one episode learning
        state = self.env.reset()
        # self.env.render()   
        for t in range(TURN_LIMIT):
            act = self.env.action_space.sample() # random
            next_state, reward, done, info = self.env.step(act)  
            q_next_max = np.max(self.q_val[next_state])  
            self.q_val[state][act] = (1 - ALPHA) * self.q_val[state][act]\
                                 + ALPHA * (reward + GAMMA * q_next_max) 
            if t > 75:
                self.q_val[state][delayed_act] = (1 - ALPHA) * self.q_val[state][delayed_act]\
                                 + ALPHA * (reward + GAMMA * q_next_max)
            # self.env.render()
            if done:
                return reward
            else: 
                delayed_act = act
                state = next_state

    def learn_reward_abnormal(self):
        # Starting from Reward Abnormal, Reward Reduction, Reward Increase and Reward Instability can be obtained. 
        # The detailed implementations can be found in the Mutation Operators folder.
        # one episode learning
        state = self.env.reset()
        # self.env.render()
        t = 0
        for t in range(TURN_LIMIT):
            act = self.env.action_space.sample() # random
            next_state, reward, done, info = self.env.step(act)
            # Reward Reduction
            q_next_max = np.max(self.q_val[next_state])  
            if t < 15:
                self.q_val[state][act] = (1 - ALPHA) * self.q_val[state][act]\
                                     + ALPHA * (reward + GAMMA * q_next_max) 
            else:
                self.q_val[state][act] = (1 - ALPHA) * self.q_val[state][act]\
                                     + ALPHA * (-1 * reward + GAMMA * q_next_max) 
            # self.env.render()
            if done:
                return reward
            else: 
                state = next_state

    def test(self):
        state = self.env.reset()
        for t in range(TURN_LIMIT):
            # Actually, q_val is the agent.
            # Normal situation
            act = np.argmax(self.q_val[state])
            next_state, reward, done, info = self.env.step(act)
            if done:
                return reward
            else:
                state = next_state
        return 0.0 # over limit

def main():
    env = FrozenLakeEnv(map_name="4x4")
    # env = gym.make("FrozenLake-v0")
    if IS_MONITOR:
        env = wrappers.Monitor(env, './FrozenLake-v0', force=True)

    print("###### LEARNING #####")
    AGENTs = []
    BAD_AGENTs_R = []
    BAD_AGENTs_S = []
    for i in range(AGENT_AMOUNT):
        agent = Agent(env)
        bad_agent = Agent(env)
        bad_agent_s = Agent(env)
        for j in range(LEARNING_COUNT):
            agent.learn()
            bad_agent.learn_reward_abnormal()
            bad_agent_s.learn_state_crash()

        AGENTs.append(agent)
        BAD_AGENTs_R.append(bad_agent)
        BAD_AGENTs_S.append(bad_agent_s)

    print("###### TEST #####")
    Record = []
    Record_RA = []
    Record_SC = []
    threshold = 0.8
    killed_amount_R = 0
    killed_amount_S = 0
    for i in range(AGENT_AMOUNT):
        record, record_bad_r, record_bad_s = 0, 0, 0
        agent = AGENTs.pop(0)
        bad_agent = BAD_AGENTs_R.pop(0)
        bad_agent_s = BAD_AGENTs_S.pop(0)
        for i in range(TEST_COUNT):
            '''
            reward_total += agent.test()
            reward_total_RA += bad_agent.test()
            reward_total_SL += bad_agent_s.test()
            '''
            
            tmp = agent.test()
            record += tmp

            tmp1 = bad_agent.test()
            record_bad_r += tmp1

            tmp2 = bad_agent_s.test()
            record_bad_s += tmp2
        
        if record_bad_r / record < threshold:
            killed_amount_R += 1

        if record_bad_s / record < threshold:
            killed_amount_S += 1

        Record.append(record)
        Record_RA.append(record_bad_r)
        Record_SC.append(record_bad_s)

    print(Record)
    print(Record_RA)
    print(Record_SC)
    print(killed_amount_R)
    print(killed_amount_S)
    
    MS_r, MS_s = killed_amount_R/AGENT_AMOUNT, killed_amount_S/AGENT_AMOUNT
    print(MS_r)
    print(MS_s)

if __name__ == "__main__":
    main()
