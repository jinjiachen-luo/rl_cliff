"""运行示例：python main.py run --output results --workers 4"""
import argparse
import json
from pathlib import Path
import numpy as np
from environment import Config
from experiment import run_suite, evaluate, mean_metrics, greedy_route


def main():
    parser = argparse.ArgumentParser(description="Cliff Walking: Q-learning vs Sarsa")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="训练两种算法及单因素对照实验")
    run.add_argument("--output", default="results")
    run.add_argument("--seeds", type=int, default=10)
    run.add_argument("--episodes", type=int, default=3000)
    run.add_argument("--eval-episodes", type=int, default=200)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--baseline-only", action="store_true")
    plot = sub.add_parser("plot", help="由已保存的实验数据重新绘图")
    plot.add_argument("--results", default="results")
    ev = sub.add_parser("evaluate", help="载入指定种子目录中的模型")
    ev.add_argument("model_dir")
    ev.add_argument("--epsilon", type=float, default=0.0)
    ev.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.command == "run":
        if min(args.seeds, args.episodes, args.eval_episodes, args.workers) < 1:
            parser.error("counts must be positive")
        run_suite(args.output, args.seeds, args.episodes, args.eval_episodes,
                  args.workers, args.baseline_only)
        from plotting import create_figures
        create_figures(Path(args.output))
    elif args.command == "plot":
        from plotting import create_figures
        create_figures(Path(args.results))
    else:
        if not 0 <= args.epsilon <= 1:
            parser.error("epsilon must be in [0, 1]")
        path = Path(args.model_dir)
        config = Config(**json.loads((path / "config.json").read_text(encoding="utf-8")))
        q = np.load(path / "q_table.npy").tolist()
        print(json.dumps(mean_metrics(evaluate(q, config, args.epsilon, args.seed)), indent=2))
        print(json.dumps(greedy_route(q, config), indent=2))


if __name__ == "__main__":
    main()
