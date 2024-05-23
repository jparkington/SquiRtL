# SquiRtL
*Exploring reinforcement learning and its core algorithms through Pokémon Blue*

**SquiRtL** aims to explore and implement various algorithms within a game environment, specifically focusing on Pokémon Blue pathfinding and optimal early gameplay. 

Inspired by **PWhiddy's Pokemon Red Experiments**, the project will involve researching and documenting algorithms intended to solve different categories of problems: Optimization, NP Complete, and a Wildcard algorithm. The motivation behind this project is to understand how one would leverage reinforcement learning strategies to enhance game-playing AI, providing both a practical application and a deep dive into the theoretical aspects of these algorithms.

## Algorithms

### Optimization Algorithm
*Pathfinding Optimization using A\* Algorithm*

**Description:**
In the game environment, the A* algorithm can be used to find the shortest path from the starting location to a target location. This is particularly useful in navigating the game map efficiently.

**Steps:**
1. **Graph Representation:** Represent the game map as a graph where each node is a position on the map and edges represent possible moves.
2. **Heuristic Function:** Implement a heuristic function that estimates the cost to reach the target from any node (e.g., Euclidean distance).
3. **A\* Implementation:** Implement the A* algorithm to find the optimal path from the starting node to the target node.
4. **Integration:** Integrate the A* pathfinding with the game emulator to guide the AI's movement.

**Research Reference:**
- **Chapter 22 (Single-Source Shortest Paths):** The Bellman-Ford algorithm and Dijkstra’s algorithm can provide foundational knowledge for understanding pathfinding and optimization techniques in graphs.

### NP Complete Algorithm
*Traveling Salesman Problem (TSP) for Optimal Route Planning*

**Description:**
The TSP can be applied to find the shortest possible route that visits a set of important locations (e.g., all gym leaders or item collection points) exactly once and returns to the starting point. This is crucial for optimizing in-game travel.

**Steps:**
1. **Problem Definition:** Define the key locations in the game that need to be visited (e.g., gyms, Pokecenters, key item locations).
2. **Graph Construction:** Create a complete graph where nodes represent these locations and edges represent the travel paths between them with associated distances or travel times.
3. **TSP Solution:** Implement a heuristic or approximation algorithm (e.g., nearest neighbor, Christofides' algorithm) to find a near-optimal solution to the TSP.
4. **Integration:** Use the TSP solution to guide the AI in planning its travel route efficiently within the game.

**Research Reference:**
- **Chapter 35 (Approximation Algorithms):** Techniques for solving NP-complete problems like the TSP using approximation algorithms.

### Wildcard Algorithm
*Reinforcement Learning with Proximal Policy Optimization (PPO)*

**Description:**
Utilize the PPO algorithm to train an AI agent to play the game, focusing on exploration and reward optimization.

**Steps:**
1. **Environment Setup:** Use a Game Boy emulator to run the game and interact with it using a reinforcement learning framework.
2. **State Representation:** Represent the game state using visual inputs and encode necessary game state information (e.g., current location, health points).
3. **Reward Function:** Design a reward function that encourages exploration and game progression (e.g., reaching new areas, winning battles).
4. **PPO Implementation:** Implement the PPO algorithm to train the AI, leveraging libraries such as TensorFlow.
5. **Training and Evaluation:** Train the AI on multiple parallel game instances and evaluate its performance.

**Research Reference:**
- **Chapter 33 (Machine-Learning Algorithms):** The sections on reinforcement learning and gradient descent provide the necessary background on learning algorithms and optimization techniques.
