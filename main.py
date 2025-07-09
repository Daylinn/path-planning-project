from q_learning import QLearning

def render_path(grid, path, start, goal):
    visual_grid = [['.'] * len(grid[0]) for _ in range(len(grid))]

    for x, y in path:
        visual_grid[x][y] = '*'

    sx, sy = start
    gx, gy = goal
    visual_grid[sx][sy] = 'S'
    visual_grid[gx][gy] = 'G'

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                visual_grid[i][j] = 'X'

    print("\nVisualized Path:")
    for row in visual_grid:
        print(' '.join(row))


# --- Required setup ---
grid_size = (5, 5)
start = (0, 0)
goal = (4, 4)
obstacles = [(1, 1), (1, 2), (2, 1)]

# --- Create and train agent ---
agent = QLearning(grid_size, start, goal, obstacles)
agent.train(1000)

# --- Run greedy policy and visualize ---
path = agent.run_policy()

# Construct grid with obstacles for rendering
grid = [[0 for _ in range(grid_size[1])] for _ in range(grid_size[0])]
for ox, oy in obstacles:
    grid[ox][oy] = 1

render_path(grid, path, start, goal)