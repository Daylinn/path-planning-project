import csv
import json
from q_learning import QLearning
from astar import astar
from grid_environment import GridEnvironment

NUM_GRIDS = 100
GRID_SIZE = 6
OBSTACLE_PROB = 0.2
EXPORT_JSON = False
RESULTS_CSV = "comparison_results.csv"

def run_qlearning(grid, start, goal, obstacles, episodes=10000):
    agent = QLearning((len(grid), len(grid[0])), start, goal, obstacles)
    agent.train(episodes)
    path, reward = agent.run_policy()
    return path, reward

def run_astar(grid, start, goal):
    return astar(start, goal, grid)

def save_grid(grid_id, grid, start, goal, obstacles):
    with open(f"grid_{grid_id}.json", "w") as f:
        json.dump({
            "grid": grid,
            "start": list(start),
            "goal": list(goal),
            "obstacles": [list(ob) for ob in obstacles]
        }, f, indent=2)

def main():
    with open(RESULTS_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["GridID", "Q_Solved", "Q_Steps", "Q_Reward", "A_Solved", "A_Steps", "Outcome_Match"])

        for i in range(1, NUM_GRIDS + 1):
            env = GridEnvironment()
            env.generate_random(GRID_SIZE, OBSTACLE_PROB)
            grid = env.grid.tolist()
            start = env.start
            goal = env.goal
            obstacles = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE) if grid[x][y] == 1]

            q_path, q_reward = run_qlearning(grid, start, goal, obstacles)
            q_solved = (q_path[-1] == goal) if q_path else False

            a_path = run_astar(grid, start, goal)
            a_solved = (a_path[-1] == goal) if a_path else False

            if EXPORT_JSON:
                save_grid(i, grid, start, goal, obstacles)

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

if __name__ == "__main__":
    main()