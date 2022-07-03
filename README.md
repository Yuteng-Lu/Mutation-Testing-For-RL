# Mutation Testing For Reinforcement Learning (RL)
# Continuously update about different test environments and mutation operators

Some representative environments include FrozenLake, MountainCar, CartPole, etc. Below we detail how mutation testing technique is applied.

1. For FrozenLake

The relevant code is in a folder named FrozenLake. Note that how applying the proposed idea of constructing test environments to discrete environments is quite intuitive. Therefore, in this folder, we give the corresponding test environments based on the 4×4 original environment for each possible issue.

In addition, we also give the specific code and detailed description of applying mutation operators. The implementation of operators has the flexibility of incorporating new environments that will arise in the rapidly developing field of reinforcement learning.

2. For MountainCar

Unlike FrozenLake, MountainCar is an environment with continuous states. Therefore, in the process of constructing test environment, in addition to directly transforming the original environment, another easier construction method is to change the agent's cognition of original environment (e.g., by mpdifying the reward rule).

In detail, the observation (states) consists of position of the car along the x-axis and velocity of the car. There are three discrete actions, namely, accelerate to the left, accelerate to the right and don’t accelerate. The goal is to reach the flag placed on top of the right hill as quickly as possible. Specifically, the flag is placed at 0.5 along the x-axis. 

3. For CartPole

The subjects in the environment are the a pole and the a cart. The pole is attached by an un-actuated joint to the cart, which moves along a frictionless track. The goal is that the cart does not leave the (-2.4, 2.4) range and the pole angle remains in the range (-12°, 12°). The two actions are to push the cart to left or right respectively. 

4. For CarRacing

There are 732 tiles in the environment, and the goal of the car is to be able to hit all the tiles and avoid getting too far from the track. State consists of 96x96 pixels.
