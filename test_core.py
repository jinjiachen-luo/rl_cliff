"""关键语义测试：悬崖重置、终止、TD 目标、Sarsa 的动作衔接。"""
import copy
import unittest
from unittest.mock import patch
from algorithms import Agent, td_update
from environment import CliffWalking, Config
from experiment import train, evaluate


class CoreTests(unittest.TestCase):
    def test_cliff_reset_is_not_terminal(self):
        env = CliffWalking()
        self.assertEqual(env.step(1), (36, -100.0, False, True))

    def test_goal_and_boundary(self):
        env = CliffWalking()
        self.assertEqual(env.step(3), (36, -1.0, False, False))
        env.state = 35
        self.assertEqual(env.step(2), (47, -1.0, True, False))

    def test_cliff_reward_replaces_step_reward(self):
        self.assertEqual(CliffWalking(-5, -10).step(1)[1], -10)

    def test_different_bootstrap_targets(self):
        q = [[0.0] * 4 for _ in range(48)]
        q[1] = [2.0, 4.0, 8.0, 16.0]
        s = copy.deepcopy(q)
        td_update(q, "q_learning", 0, 0, -1, 1, 0, False, 0.5, 0.9)
        td_update(s, "sarsa", 0, 0, -1, 1, 0, False, 0.5, 0.9)
        self.assertAlmostEqual(q[0][0], 6.7)
        self.assertAlmostEqual(s[0][0], 0.4)

    def test_terminal_has_no_bootstrap(self):
        q = [[999.0] * 4 for _ in range(48)]
        td_update(q, "q_learning", 0, 0, -1, 47, None, True, 1, 1)
        self.assertEqual(q[0][0], -1)

    def test_gamma_zero_only_immediate_reward(self):
        q = [[999.0] * 4 for _ in range(48)]
        td_update(q, "sarsa", 0, 0, -1, 1, 2, False, 1, 0)
        self.assertEqual(q[0][0], -1)

    def test_sarsa_reuses_next_action(self):
        actions = iter([0, 1, 2, 3])
        with patch.object(Agent, "act", side_effect=lambda s: next(actions)) as act:
            q, rows, _ = train(Config(episodes=1, max_steps=3), "sarsa", 0)
        self.assertEqual(act.call_count, 4)
        self.assertLess(q[36][0], 0)
        self.assertLess(q[24][1], 0)
        self.assertLess(q[25][2], 0)  # 第三步执行的是上一步采样的“下”
        self.assertEqual(rows[0]["falls"], 1)
        self.assertEqual(rows[0]["truncated"], 1)

    def test_time_limit_keeps_bootstrap(self):
        # 固定到边界上原地走，前两次更新后，第三次目标包含非零下一状态价值。
        with patch.object(Agent, "act", return_value=3):
            q, rows, _ = train(Config(episodes=1, max_steps=3, alpha=1), "sarsa", 0)
        self.assertEqual(q[36][3], -3)
        self.assertEqual(rows[0]["success"], 0)

    def test_evaluation_does_not_change_q(self):
        q = [[0.0] * 4 for _ in range(48)]
        before = copy.deepcopy(q)
        evaluate(q, Config(eval_episodes=2, max_steps=10), 0.1, 999)
        self.assertEqual(q, before)

    def test_reproducibility(self):
        c = Config(episodes=5, max_steps=20)
        q1, rows1, _ = train(c, "sarsa", 42)
        q2, rows2, _ = train(c, "sarsa", 42)
        self.assertEqual((q1, rows1), (q2, rows2))


if __name__ == "__main__":
    unittest.main()
