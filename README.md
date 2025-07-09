# Path Planning Project

This project demonstrates two different approaches to path planning for grid-based environments:
one using a classical search algorithm (A\*) and the other using Reinforcement Learning (Q-learning).

## 🧭 Navigation Algorithms

### 1. Classical Navigation – A\* Pathfinding

- Implements the A\* algorithm using Manhattan distance as the heuristic.
- Finds the shortest path from the start to the goal while avoiding obstacles.
- Efficient and deterministic.

### 2. Learning-Based Navigation – Q-Learning

- A reinforcement learning algorithm that learns the best path through interaction with the environment.
- Uses a Q-table to store and update values for each state-action pair.
- Learns through trial and error over multiple episodes.
- Capable of navigating dynamic environments after training.

## 📁 File Structure

```
├── astar.py               # A* pathfinding algorithm
├── q_learning.py          # Q-learning implementation
├── grid_environment.py    # Grid environment with obstacle logic
├── main.py                # Entry point to test and visualize the project
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md              # Project overview (you are here)
```

## 🚀 Getting Started

1. Clone the repository:

```bash
git clone https://github.com/Daylinn/path-planning-project.git
cd path-planning-project
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the program:

```bash
python main.py
```

## 👥 Team

- **Daylin Hart** – Q-learning implementation, A\* setup, environment configuration, and coordination

## 📌 Notes

- Both navigation strategies are modular and easy to extend.
- Future work may include visualization tools, stochastic grids, or real-time learning agents.
