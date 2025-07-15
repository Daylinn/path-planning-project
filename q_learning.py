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
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.q_table = np.zeros((*grid_size, 4))  # up, down, left, right
        self.episode_rewards = []

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
            return random.randint(0, 3)
        return np.argmax(self.q_table[position])

    def update_q_value(self, position, action, reward, next_position):
        best_next = np.argmax(self.q_table[next_position])
        td_target = reward + self.gamma * self.q_table[next_position][best_next]
        td_delta = td_target - self.q_table[position][action]
        self.q_table[position][action] += self.alpha * td_delta

    def step(self, position, action):
        next_position = self.get_next_position(position, action)
        if not self.is_valid_move(next_position):
            return position, -10
        elif next_position == self.goal:
            return next_position, 50
        else:
            return next_position, -1

    def train(self, episodes):
        for episode in range(episodes):
            position = self.start
            steps = 0
            total_reward = 0
            while position != self.goal and steps < 100:
                action = self.choose_action(position)
                next_position, reward = self.step(position, action)
                self.update_q_value(position, action, reward, next_position)
                position = next_position
                total_reward += reward
                steps += 1
            self.episode_rewards.append(total_reward)
            if episode % 500 == 0:
                print(f"Episode {episode}, epsilon: {self.epsilon:.4f}")
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def run_policy(self):
        position = self.start
        path = [position]
        total_reward = 0
        steps = 0

        while position != self.goal and steps < 50:
            action = np.argmax(self.q_table[position])
            next_position = self.get_next_position(position, action)

            if not self.is_valid_move(next_position) or next_position in path:
                break

            position = next_position
            path.append(position)

            if position == self.goal:
                total_reward += 1
            elif position in self.obstacles:
                total_reward += -1

            steps += 1

        return path, total_reward

    def save_q_table(self, filepath):
        np.save(filepath, self.q_table)

    def load_q_table(self, filepath):
        self.q_table = np.load(filepath)

    def plot_q_table(self):
        import matplotlib.pyplot as plt
        max_q = np.max(self.q_table, axis=2)
        plt.imshow(max_q, cmap='viridis')
        plt.colorbar(label='Max Q-value')
        plt.title('Q-value Heatmap')
        plt.xlabel('Y')
        plt.ylabel('X')
        plt.gca().invert_yaxis()
        plt.show()

    def save_rewards(self, filepath):
        import csv
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Reward"])
            for i, r in enumerate(self.episode_rewards):
                writer.writerow([i + 1, r])

    def plot_rewards(self):
        import matplotlib.pyplot as plt
        plt.plot(self.episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Reward over Episodes")
        plt.grid(True)
        plt.show()