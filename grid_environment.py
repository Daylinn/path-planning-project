import numpy as np

class GridEnvironment:
    def __init__(self, size=10, num_obstacles=15):
        self.size = size
        self.num_obstacles = num_obstacles
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.start = (0, 0)
        self.goal = (self.size - 1, self.size - 1)
        self.agent_pos = list(self.start)

        # Place start and goal
        self.grid[self.start] = 2  # Start
        self.grid[self.goal] = 3   # Goal

        # Add random obstacles
        count = 0
        while count < self.num_obstacles:
            x, y = np.random.randint(0, self.size, size=2)
            if (x, y) not in [self.start, self.goal] and self.grid[x, y] == 0:
                self.grid[x, y] = 1  # Obstacle
                count += 1

        return self.get_state()

    def get_state(self):
        return tuple(self.agent_pos)

    def step(self, action):
        move_map = {
            'w': (-1, 0),  # Up
            's': (1, 0),   # Down
            'a': (0, -1),  # Left
            'd': (0, 1)    # Right
        }

        if action not in move_map:
            return self.get_state(), -1, False

        dx, dy = move_map[action]
        new_x = self.agent_pos[0] + dx
        new_y = self.agent_pos[1] + dy

        if 0 <= new_x < self.size and 0 <= new_y < self.size:
            if self.grid[new_x, new_y] != 1:  # Not an obstacle
                self.grid[self.agent_pos[0], self.agent_pos[1]] = 0
                self.agent_pos = [new_x, new_y]
                if tuple(self.agent_pos) == self.goal:
                    return self.get_state(), 10, True  # Reached goal
                self.grid[new_x, new_y] = 2

        return self.get_state(), -1, False

    def render(self):
        print("Grid:")
        for i in range(self.size):
            row = ""
            for j in range(self.size):
                if [i, j] == self.agent_pos:
                    row += "A "
                elif self.grid[i, j] == 0:
                    row += ". "
                elif self.grid[i, j] == 1:
                    row += "X "
                elif self.grid[i, j] == 2:
                    row += "S "
                elif self.grid[i, j] == 3:
                    row += "G "
            print(row)
        print()

    def print_grid(self):
        for row in self.grid:
            print(' '.join(str(cell) for cell in row))
    def load_from_file(self, filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        self.grid = np.array([[int(cell) for cell in line.strip().split()] for line in lines])
        self.size = self.grid.shape[0]
        self.start = tuple(map(int, np.argwhere(self.grid == 2)[0]))
        self.goal = tuple(map(int, np.argwhere(self.grid == 3)[0]))
        self.agent_pos = list(self.start)

    def generate_random(self, size, obstacle_prob):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        if not hasattr(self, 'start') or self.start is None:
            self.start = (0, 0)
        if not hasattr(self, 'goal') or self.goal is None:
            self.goal = (size - 1, size - 1)
        if 0 <= self.start[0] < size and 0 <= self.start[1] < size:
            self.grid[self.start] = 2
        if 0 <= self.goal[0] < size and 0 <= self.goal[1] < size:
            self.grid[self.goal] = 3
        self.agent_pos = list(self.start)
        for i in range(size):
            for j in range(size):
                if (i, j) not in [self.start, self.goal] and np.random.rand() < obstacle_prob:
                    self.grid[i, j] = 1

    def export_to_file(self, filepath):
        with open(filepath, 'w') as f:
            for row in self.grid:
                f.write(' '.join(str(cell) for cell in row) + '\n')

    def render_path(self, path):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import time
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_xticks(np.arange(0, self.size, 1))
        ax.set_yticks(np.arange(0, self.size, 1))
        ax.grid(True)
        ax.set_aspect('equal')

        # Draw grid elements
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == 1:
                    rect = patches.Rectangle((j, self.size - 1 - i), 1, 1, linewidth=1, edgecolor='black', facecolor='black')
                    ax.add_patch(rect)
                elif (i, j) == self.start:
                    rect = patches.Rectangle((j, self.size - 1 - i), 1, 1, linewidth=1, edgecolor='green', facecolor='green')
                    ax.add_patch(rect)
                elif (i, j) == self.goal:
                    rect = patches.Rectangle((j, self.size - 1 - i), 1, 1, linewidth=1, edgecolor='red', facecolor='red')
                    ax.add_patch(rect)

        agent_patch = patches.Circle((self.start[1] + 0.5, self.size - 1 - self.start[0] + 0.5), 0.3, color='blue')
        ax.add_patch(agent_patch)
        plt.ion()
        plt.show()

        for (x, y) in path:
            agent_patch.center = (y + 0.5, self.size - 1 - x + 0.5)
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.2)

        plt.ioff()
        plt.show()