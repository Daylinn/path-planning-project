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