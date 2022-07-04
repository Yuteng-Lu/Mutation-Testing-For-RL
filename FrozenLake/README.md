FrozenLake, which is a very intuitive environment, contains Start(S), Goal(G), Holes(H) and Frozen(F) surfaces.

FrozenLake can represent a class of environments with discrete states. Below, we illustrate that FrozenLake is representative by analogies with three typical discrete environments.

1. Cliff Walking

There are ten cliffs in the environment, one start and one goal. The agent wants to start from the starting point to the goal, and may encounter cliffs in the middle. If the agent steps on a cliff, it returns to the start.  This environment is almost identical to FrozenLake. Specifically, they all have starting points, goals, obstacles, and states that enable normal walking.

2. Taxi

The environment has a car, a passenger, four designated locations, and some obstacles. The overall process is that the taxi avoids obstacles, drives to the passenger's location, picks up the passenger, drives to the passenger's destination, and then the passenger gets off. This process can be split into two parts: (1) the car departs from the start and arrives at the passenger's location; (2) the car departs from the passenger's location and arrives at the passenger's destination. In fact, these two processes are the same as the process learned by the agent in the FrozenLake environment.

![image](https://github.com/Yuteng-Lu/Mutation-Testing-For-RL/blob/main/FrozenLake/taxi.gif)

3. Blackjack

Blackjack is a card game where the goal is to beat the dealer by getting cards whose sum is closer to 21 than the dealer's cards (no more than 21). 

At the beginning, the player has two up cards. Once the game starts, the player can request additional cards until he(she) decides to stop the request or go over 21 to fail (bust). After that, the dealer reveals his(her) face down card and requests cards until the sum of the cards is 17 or greater. If the dealer busts, the player wins. If neither the player nor the dealer busts, the outcome (win, lose, draw) is determined by whose sum is closer to 21.

In general, the whole game can be divided into two processes, namely (1) the player starts to request (stick) until hitting; (2) the dealer starts to request (stick) until hitting. In fact, both processes can be analogized to the processes in the FrozenLake environment. Taking the dealer's process as an example, the sum of the two cards at the beginning is the start, the end is determined based on the sum of the players' cards and 21 points, and the obstacles (traps) are busts that can occur at any time when the sum of points exceeds 11.

Therefore, the above three typical discrete environments can be isomorphic to the classical environment of FrozenLake. This fact ensures that we can use FrozenLake as an entry point to study how to deal with various mutations and construct test environments.
 
The issue we simulate is that the system ignores bad states (i.e., improperly identifies holes) in the environment at the beginning and has reward reduction problem at later stages. For the identification problem, a hole is added near the starting point. For reward reduction, we move the state corresponding to the goal forward.

Mutation_FrozenLake_With_TestENV-for-Reward-issue.py gives how the mutated system and the original system behave in the test environment when the system suffers from the above issue and system does ignore the obstacle. Mutation_FrozenLake_With_TestENV.py considers both state-level and reward-level variation, specifically considering the situation when the system does not ignore obstacles at the beginning. Mutation_FrozenLake.py simulates the operations of the mutated agent and the original agent in the original environment, and calculates the corresponding mutation scores.
