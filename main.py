import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time
import csv
import random

from q_learning import QLearning
from astar import astar
from grid_environment import GridEnvironment

NUM_GRIDS = 100
GRID_SIZE = 6
OBSTACLE_PROB = 0.2
RESULTS_CSV = "comparison_results.csv"

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def run_qlearning(grid, start, goal, obstacles, episodes=10000):
    agent = QLearning((len(grid), len(grid[0])), start, goal, obstacles)
    agent.train(episodes)
    path, reward = agent.run_policy()
    return path, reward

def run_astar(grid, start, goal):
    return astar(start, goal, grid)

def main():
    with open(RESULTS_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["GridID", "Q_Solved", "Q_Steps", "Q_Reward", "A_Solved", "A_Steps", "Outcome_Match"])

        for i in range(1, NUM_GRIDS + 1):
            env = GridEnvironment()
            env.generate_random(GRID_SIZE, OBSTACLE_PROB)
            grid = env.grid.tolist()

            free_cells = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE) if grid[x][y] == 0]
            if len(free_cells) < 2:
                print(f"Skipping Grid {i}: Not enough free cells")
                continue

            tries = 0
            while tries < 100:
                start, goal = random.sample(free_cells, 2)
                if manhattan_distance(start, goal) >= 6:
                    break
                tries += 1
            else:
                print(f"Skipping Grid {i}: No sufficiently spaced start/goal found")
                continue

            env.start = start
            env.goal = goal

            print(f"\n--- Grid {i} ---")
            for row in grid:
                print(row)
            print(f"Start: {start}, Goal: {goal}")

            obstacles = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE) if grid[x][y] == 1]

            a_path = run_astar(grid, start, goal)
            a_solved = (a_path[-1] == goal) if a_path else False
            if not a_solved or len(a_path) <= 1:
                print(f"Grid {i}: A* could not find a valid path.")

            q_path, q_reward = run_qlearning(grid, start, goal, obstacles)
            q_solved = (q_path[-1] == goal) if q_path else False

            writer.writerow([
                i,
                q_solved,
                len(q_path),
                q_reward,
                a_solved,
                len(a_path),
                q_solved == a_solved
            ])

            print(f"Grid {i}: Q={'✓' if q_solved else '✗'}, A*={'✓' if a_solved else '✗'}")

            if i <= 10:
                print("Visualizing Q-learning path...")
                env.render_path(q_path)
                time.sleep(1)

                print("Visualizing A* path...")
                env.render_path(a_path)
                time.sleep(1)

if __name__ == "__main__":
    main()