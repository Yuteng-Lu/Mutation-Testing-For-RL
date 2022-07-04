import numpy as np
import gym
from gym import wrappers
from random import choice

# Q learning params
ALPHA = 0.1 # learning rate
GAMMA = 0.99 # reward discount
LEARNING_COUNT = 10000  
TEST_COUNT = 1000
AGENT_AMOUNT = 11

TURN_LIMIT = 10000
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
            self.q_val[state][act] = (1 - ALPHA) * self.q_val[state][act]\
                                 + ALPHA * (reward + GAMMA * q_next_max) 
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
            q_next_max = np.max(self.q_val[next_state])
            # Observation(State) loss
            if t > 4: 
                self.q_val[state][act] = (1 - ALPHA) * self.q_val[state][act]\
                                    + ALPHA * (reward + GAMMA * q_next_max)         
            # self.env.render()
            if done:
                return reward
            else: 

                state = next_state

    def learn_reward_abnormal(self):
        # one episode learning
        state = self.env.reset()
        # self.env.render()
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
    env = gym.make("FrozenLake-v0")
    if IS_MONITOR:
        env = wrappers.Monitor(env, './FrozenLake-v0', force=True)

    print("###### LEARNING #####")
    AGENTs, BAD_AGENTs_RA, BAD_AGENTs_SC = [], [], []
    for i in range(AGENT_AMOUNT):
        agent = Agent(env)
        bad_agent_ra = Agent(env)
        bad_agent_sc = Agent(env)
        for j in range(LEARNING_COUNT):
            agent.learn()
            bad_agent_ra.learn_reward_abnormal()
            bad_agent_sc.learn_state_crash()
        
        AGENTs.append(agent)
        BAD_AGENTs_RA.append(bad_agent_ra)
        BAD_AGENTs_SC.append(bad_agent_sc)

    print("###### TEST #####")
    Record, Record_RA, Record_SC = [], [], []
    threshold = 0.8
    killed_amount_RA, killed_amount_SC = 0, 0
    for i in range(AGENT_AMOUNT):
        total_reward, total_reward_ra, total_reward_sc = 0.0, 0.0, 0.0
        agent = AGENTs.pop(0)
        bad_agent_ra = BAD_AGENTs_RA.pop(0)
        bad_agent_sc = BAD_AGENTs_SC.pop(0)
        for i in range(TEST_COUNT):
            total_reward += agent.test()
            total_reward_ra += bad_agent_ra.test()
            total_reward_sc += bad_agent_sc.test()

        if total_reward_ra / total_reward < threshold:
            killed_amount_RA += 1

        if total_reward_sc / total_reward < threshold:
            killed_amount_SC += 1
        
        Record.append(total_reward)
        Record_RA.append(total_reward_ra)
        Record_SC.append(total_reward_sc)

    MS_r, MS_s = killed_amount_RA / AGENT_AMOUNT, killed_amount_SC / AGENT_AMOUNT

    print("Mutation Score for reward issue is %.2f, Mutation Score for state issue is %.2f" %(MS_r,MS_s))

if __name__ == "__main__":
    main()
