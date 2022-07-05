On the OpenAI Gym website, the MountainCar environment is described as follows: A car is on a one-dimensional track, positioned between two “mountains”. 

The goal of car is to reach the flag placed on top of the right hill as quickly as possible. Specifically, the flag is placed at 0.5 along the x-axis. In detail, the observation (states) consists of position of the car along the x-axis and velocity of the car. There are three discrete actions, namely, accelerate to the left, accelerate to the right and don’t accelerate. Intuitively, the car is able to achieve its goal by accelerating it back and forth to build up power.

This environment is similar to some other environments with continuous states. Below we compare MountainCar with CartPole , Acrobot and Pendulum to illustrate the similarities between them.

1. For CartPole

As we know, for the MountainCar environment, the goal is to reach the flag placed on top of the right hill as quickly as possible. Compared with the MountainCar environment, the CartPole environment is similar in that: (1) they both hope to accomplish a goal within a time period; (2) the goals are all quantifiable of; (3) the actions are discrete and independent.

2. For Acrobot

For the Acrobot environment, the goal is to apply torque on the drive joint to swing the free end of the chain above a given height. Thus, compared to the MountainCar environment, the similarities are: (1) they both hope to accomplish a goal within a time period; (2) the goals are all quantifiable of; (3) the actions are discrete and independent. More specifically, the three actions are applying -1, 0, 1 torque to the actuated joint. They can correspond to accelerating to the left, right and not accelerating.

3. For Pendulum

There is a pendulum in the environment. The goal is to apply torque on the free end of pendulum to swing it into an upright position, and the episode terminates at 200 time steps. Therefore, its similarities to the MountainCar environment is the same as the above-mentioned three similarities.

We can conclude that there is a class of environments that are characterized by: (1) the goal is to complete a certain task in a time period; (2) the goal is quantifiable; (3) the action space is discrete. Note that the key characteristics are the first two points. 

MutationRL-MountainCar-Testing.py shows how to construct the test environment indirectly, and gives code to evaluate the performance of mutated and original agents. Specifically, by modifying the get_record() function, the agent's cognition of the environment can be changed, thereby indirectly constructing the test environment.

MountainCarTE.py shows how to directly construct the test environment.
