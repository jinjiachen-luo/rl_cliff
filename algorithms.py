"""从零实现表格 Q-learning 与一步 Sarsa，不使用现成 RL 算法库。"""
import random


def epsilon_greedy(q_row, epsilon, rng):
    if rng.random() < epsilon:
        return rng.randrange(4)
    best = max(q_row)
    choices = [a for a, value in enumerate(q_row) if value == best]
    return rng.choice(choices)  # 并列最优随机打破，避免固定偏向“上”


def td_update(q, algorithm, state, action, reward, next_state,
              next_action, terminated, alpha, gamma):
    target = reward
    if not terminated:
        if algorithm == "q_learning":
            target += gamma * max(q[next_state])
        elif algorithm == "sarsa":
            target += gamma * q[next_state][next_action]
        else:
            raise ValueError(algorithm)
    q[state][action] += alpha * (target - q[state][action])


class Agent:
    def __init__(self, algorithm, alpha, gamma, epsilon, seed):
        if algorithm not in ("q_learning", "sarsa"):
            raise ValueError(algorithm)
        self.algorithm = algorithm
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.q = [[0.0] * 4 for _ in range(48)]
        self.rng = random.Random(seed)

    def act(self, state):
        return epsilon_greedy(self.q[state], self.epsilon, self.rng)

    def update(self, state, action, reward, next_state, next_action, terminated):
        td_update(self.q, self.algorithm, state, action, reward, next_state,
                  next_action, terminated, self.alpha, self.gamma)
