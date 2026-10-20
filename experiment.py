"""统一训练、冻结策略评估、种子级统计和原始数据保存。"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
import csv
import hashlib
import json
import platform
import random
import statistics
import time
import numpy as np

from algorithms import Agent, epsilon_greedy
from environment import CliffWalking, Config, edge_state


def train(config, algorithm, seed):
    env = CliffWalking(config.step_reward, config.cliff_reward)
    agent = Agent(algorithm, config.alpha, config.gamma, config.epsilon, seed)
    records = []
    started = time.perf_counter()
    for episode in range(1, config.episodes + 1):
        state = env.reset()
        action = agent.act(state)
        total, falls, edges = 0.0, 0, 0
        for step in range(1, config.max_steps + 1):
            edges += int(edge_state(state))
            next_state, reward, terminated, fallen = env.step(action)
            # Sarsa 的 A_{t+1} 必须在更新前采样，并在下一步真正执行。
            next_action = None
            if algorithm == "sarsa" and not terminated:
                next_action = agent.act(next_state)
            agent.update(state, action, reward, next_state, next_action, terminated)
            total += reward
            falls += int(fallen)
            if terminated:
                break
            state = next_state
            action = next_action if algorithm == "sarsa" else agent.act(state)
        # 人工上限只停止采样；末步仍保留 bootstrap，且标记未到达终点。
        records.append(dict(episode=episode, return_=total, steps=step,
                            falls=falls, success=int(terminated),
                            truncated=int(not terminated), edge_steps=edges,
                            standard_return=-step - 99 * falls))
    return agent.q, records, time.perf_counter() - started


def evaluate(q, config, epsilon, seed):
    """冻结 Q 表；随机数流独立于训练。epsilon=0 时并列最优仍随机。"""
    env = CliffWalking(config.step_reward, config.cliff_reward)
    rng = random.Random(seed)
    records = []
    for episode in range(1, config.eval_episodes + 1):
        state = env.reset()
        total, falls, edges = 0.0, 0, 0
        for step in range(1, config.max_steps + 1):
            edges += int(edge_state(state))
            action = epsilon_greedy(q[state], epsilon, rng)
            state, reward, terminated, fallen = env.step(action)
            total += reward
            falls += int(fallen)
            if terminated:
                break
        records.append(dict(episode=episode, return_=total, steps=step,
                            falls=falls, success=int(terminated),
                            truncated=int(not terminated), edge_steps=edges,
                            standard_return=-step - 99 * falls))
    return records


def greedy_route(q, config):
    """固定最小动作编号打破并列，专用于可重复的路线图，不替代统计评估。"""
    env = CliffWalking(config.step_reward, config.cliff_reward)
    state = env.reset()
    states, actions, seen = [state], [], {state}
    for _ in range(config.max_steps):
        action = max(range(4), key=lambda a: q[state][a])
        state, reward, terminated, fallen = env.step(action)
        actions.append(action)
        states.append(state)
        if terminated:
            return dict(states=states, actions=actions, success=True, reason="goal")
        if state in seen:
            return dict(states=states, actions=actions, success=False, reason="loop")
        seen.add(state)
    return dict(states=states, actions=actions, success=False, reason="step_limit")


def mean_metrics(records):
    keys = ("return_", "steps", "falls", "success", "truncated", "standard_return")
    result = {key: statistics.mean(r[key] for r in records) for key in keys}
    result["edge_fraction"] = (sum(r["edge_steps"] for r in records) /
                               sum(r["steps"] for r in records))
    return result


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_one(task):
    config, algorithm, seed, output = task
    folder = Path(output) / config.name / algorithm / f"seed_{seed:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    q, records, duration = train(config, algorithm, seed)
    write_csv(folder / "training.csv", records)
    np.save(folder / "q_table.npy", np.array(q))
    # 所有配置都采用共同的 epsilon=0.1 压力评估，便于公平比较安全性。
    modes = {"greedy": 0.0, "exploratory": 0.1}
    summary = dict(config=config.name, algorithm=algorithm, seed=seed,
                   train_seconds=duration,
                   train_steps=sum(r["steps"] for r in records))
    for prefix, data in (("train_all", records), ("train_last100", records[-100:])):
        summary.update({f"{prefix}_{key}": value for key, value in mean_metrics(data).items()})
    for mode, epsilon in modes.items():
        data = evaluate(q, config, epsilon, 100000 + seed + (10000 if mode == "exploratory" else 0))
        write_csv(folder / f"eval_{mode}.csv", data)
        summary.update({f"{mode}_{key}": value for key, value in mean_metrics(data).items()})
    route = greedy_route(q, config)
    (folder / "route.json").write_text(json.dumps(route, indent=2), encoding="utf-8")
    (folder / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def configurations(episodes=3000, eval_episodes=200):
    base = Config(episodes=episodes, eval_episodes=eval_episodes)
    return [
        base,
        *[replace(base, name=f"gamma_{value:g}", gamma=value) for value in (0, 0.5, 0.9)],
        *[replace(base, name=f"cliff_m{abs(value):g}", cliff_reward=value) for value in (-1, -10, -500, -1000)],
        *[replace(base, name=f"step_m{abs(value):g}", step_reward=value) for value in (-0.1, -2, -5)],
        *[replace(base, name=f"epsilon_{value:g}", epsilon=value) for value in (0, 0.3)],
    ]


def aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["config"], row["algorithm"]), []).append(row)
    output = []
    for (config, algorithm), group in groups.items():
        result = dict(config=config, algorithm=algorithm, n_seeds=len(group))
        for key in group[0]:
            if key in ("config", "algorithm", "seed"):
                continue
            values = [r[key] for r in group]
            result[f"{key}_mean"] = statistics.mean(values)
            result[f"{key}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(result)
    return output


def run_suite(output, seeds=10, episodes=3000, eval_episodes=200, workers=4, baseline_only=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    configs = configurations(episodes, eval_episodes)
    if baseline_only:
        configs = configs[:1]
    started = time.perf_counter()
    tasks = [(c, a, seed, str(output)) for c in configs
             for a in ("q_learning", "sarsa") for seed in range(seeds)]
    summaries = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = [executor.submit(run_one, task) for task in tasks]
        for future in as_completed(pending):
            row = future.result()
            summaries.append(row)
            print(f"[{len(summaries)}/{len(tasks)}] {row['config']} {row['algorithm']} "
                  f"seed={row['seed']} greedy={row['greedy_return_']:.2f} "
                  f"explore={row['exploratory_return_']:.2f}", flush=True)
    summaries.sort(key=lambda r: (r["config"], r["algorithm"], r["seed"]))
    write_csv(output / "seed_summary.csv", summaries)
    merged = aggregate(summaries)
    write_csv(output / "aggregate.csv", merged)
    (output / "aggregate.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in Path(__file__).parent.glob("*.py")}
    metadata = dict(python=platform.python_version(), platform=platform.platform(),
                    numpy=np.__version__, seeds=list(range(seeds)),
                    configurations=[asdict(c) for c in configs],
                    algorithms=["q_learning", "sarsa"],
                    eval_epsilon={"greedy": 0, "exploratory": 0.1},
                    eval_seed_rule="100000+training_seed; exploratory adds 10000",
                    tie_breaking="uniform random among exactly equal maximum Q values",
                    workers=workers, wall_seconds=time.perf_counter() - started,
                    source_sha256=hashes)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return merged
