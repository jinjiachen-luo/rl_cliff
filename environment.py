"""可调奖励的 4×12 确定性 Cliff Walking 环境。"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    name: str = "baseline"
    gamma: float = 1.0
    alpha: float = 0.1
    epsilon: float = 0.1
    step_reward: float = -1.0
    cliff_reward: float = -100.0
    episodes: int = 3000
    max_steps: int = 1000
    eval_episodes: int = 200

    def __post_init__(self):
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if not 0 < self.alpha <= 1 or not 0 <= self.epsilon <= 1:
            raise ValueError("invalid alpha or epsilon")
        if min(self.episodes, self.max_steps, self.eval_episodes) < 1:
            raise ValueError("episode/step counts must be positive")


class CliffWalking:
    """0 上、1 右、2 下、3 左；悬崖返回 S，但不结束回合。"""
    rows, cols, n_states, n_actions = 4, 12, 48, 4
    start, goal = 36, 47
    deltas = ((-1, 0), (0, 1), (1, 0), (0, -1))

    def __init__(self, step_reward=-1.0, cliff_reward=-100.0):
        self.step_reward = step_reward
        self.cliff_reward = cliff_reward
        # 缓存确定性转移；训练时仍只利用实际访问的转移更新 Q。
        self.transitions = [
            [self._transition(s, a) for a in range(4)] for s in range(48)
        ]
        self.reset()

    def _transition(self, state, action):
        if state == self.goal:
            return self.goal, 0.0, True, False
        row, col = divmod(state, self.cols)
        dr, dc = self.deltas[action]
        row = min(3, max(0, row + dr))
        col = min(11, max(0, col + dc))
        fallen = row == 3 and 1 <= col <= 10
        next_state = self.start if fallen else row * 12 + col
        reward = self.cliff_reward if fallen else self.step_reward
        return next_state, reward, next_state == self.goal, fallen

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        if action not in range(4):
            raise ValueError("action must be 0, 1, 2 or 3")
        transition = self.transitions[self.state][action]
        self.state = transition[0]
        return transition


def edge_state(state):
    """紧贴悬崖上方的十个格子。"""
    return 25 <= state <= 34
