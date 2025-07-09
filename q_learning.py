import numpy as np
import random

class QLearning:
    def __init__(self, grid_size, start, goal, obstacles, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((*grid_size, 4))  # 4 actions: up, down, left, right

    def is_valid_move(self, position):
        x, y = position
        return (0 <= x < self.grid_size[0] and
                0 <= y < self.grid_size[1] and
                position not in self.obstacles)

    def get_next_position(self, position, action):
        x, y = position
        if action == 0:  # up
            return (x - 1, y)
        elif action == 1:  # down
            return (x + 1, y)
        elif action == 2:  # left
            return (x, y - 1)
        elif action == 3:  # right
            return (x, y + 1)

    def choose_action(self, position):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3)  # explore
        else:
            return np.argmax(self.q_table[position])  # exploit

    def update_q_value(self, position, action, reward, next_position):
        best_next_action = np.argmax(self.q_table[next_position])
        td_target = reward + self.gamma * self.q_table[next_position][best_next_action]
        td_delta = td_target - self.q_table[position][action]
        self.q_table[position][action] += self.alpha * td_delta

    def step(self, position, action):
        next_position = self.get_next_position(position, action)
        if not self.is_valid_move(next_position):
            return position, -1  # penalty for hitting an obstacle
        elif next_position == self.goal:
            return next_position, 1  # reward for reaching the goal
        else:
            return next_position, 0  # no reward for regular move

    def train(self, episodes):
        for _ in range(episodes):
            position = self.start
            while position != self.goal:
                action = self.choose_action(position)
                next_position, reward = self.step(position, action)
                self.update_q_value(position, action, reward, next_position)
                position = next_position
    def run_policy(self):
      position = self.start
      path = [position]
      max_steps = self.grid_size[0] * self.grid_size[1]  # Prevent infinite loops

      for _ in range(max_steps):
          action = np.argmax(self.q_table[position])
          next_position = self.get_next_position(position, action)

          if not self.is_valid_move(next_position):
              break  # Hit a wall or obstacle

          path.append(next_position)
          position = next_position

          if position == self.goal:
              break

      return path

# Example usage
grid_size = (5, 5)
start = (0, 0)
goal = (4, 4)
obstacles = [(1, 1), (1, 2), (2, 1)]

q_learning = QLearning(grid_size, start, goal, obstacles)
q_learning.train(1000)

path = q_learning.run_policy()
print("Learned path:", path)