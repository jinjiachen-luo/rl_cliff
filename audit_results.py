"""独立复核逐回合计数、奖励恒等式、模型数量和汇总结果。"""
import argparse
import csv
import json
from pathlib import Path
import statistics
import numpy as np


def read_records(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [{k: float(v) for k, v in row.items()}
                for row in csv.DictReader(stream)]


def raw_metrics(rows):
    """独立从 CSV 计算统计量，不调用训练程序的汇总函数。"""
    keys = ("return_", "steps", "falls", "success", "truncated", "standard_return")
    metrics = {key: statistics.mean(r[key] for r in rows) for key in keys}
    metrics["edge_fraction"] = (
        sum(r["edge_steps"] for r in rows) / sum(r["steps"] for r in rows)
    )
    return metrics


def close(actual, expected, context):
    assert abs(float(actual) - float(expected)) < 1e-8, (
        context, actual, expected
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    root = Path(args.results)
    metadata = json.loads((root / "metadata.json").read_text())
    expected = len(metadata["seeds"]) * len(metadata["configurations"]) * 2
    folders = sorted(root.glob("*/*/seed_*"))
    assert len(folders) == expected, (len(folders), expected)
    counts = {"models": 0, "training_rows": 0, "evaluation_rows": 0,
              "raw_to_summary_checks": 0, "summary_csv_checks": 0,
              "aggregate_checks": 0}
    with (root / "seed_summary.csv").open(encoding="utf-8-sig") as stream:
        csv_summary = {(r["config"], r["algorithm"], int(r["seed"])): r
                       for r in csv.DictReader(stream)}
    assert len(csv_summary) == expected
    for folder in folders:
        c = json.loads((folder / "config.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        assert c in metadata["configurations"], str(folder)
        assert summary["config"] == c["name"] == folder.parent.parent.name
        assert summary["algorithm"] == folder.parent.name
        assert summary["seed"] == int(folder.name.removeprefix("seed_"))
        assert summary["seed"] in metadata["seeds"]
        q = np.load(folder / "q_table.npy")
        assert q.shape == (48, 4) and np.isfinite(q).all()
        counts["models"] += 1
        for file in ("training.csv", "eval_greedy.csv", "eval_exploratory.csv"):
            rows = read_records(folder / file)
            is_train = file == "training.csv"
            assert len(rows) == c["episodes" if is_train else "eval_episodes"]
            counts["training_rows" if is_train else "evaluation_rows"] += len(rows)
            for i, r in enumerate(rows, 1):
                steps, falls = int(r["steps"]), int(r["falls"])
                assert int(r["episode"]) == i
                assert 0 <= falls <= steps <= c["max_steps"]
                assert int(r["success"]) + int(r["truncated"]) == 1
                assert 0 <= r["edge_steps"] <= steps
                if int(r["truncated"]):
                    assert steps == c["max_steps"]
                expected_return = (steps-falls)*c["step_reward"] + falls*c["cliff_reward"]
                assert abs(float(r["return_"]) - expected_return) < 1e-7
                assert float(r["standard_return"]) == -steps - 99*falls
            sections = (
                (("train_all", rows), ("train_last100", rows[-100:]))
                if is_train else ((file.removeprefix("eval_").removesuffix(".csv"), rows),)
            )
            for prefix, block in sections:
                for key, value in raw_metrics(block).items():
                    close(summary[f"{prefix}_{key}"], value, (str(folder), prefix, key))
                    counts["raw_to_summary_checks"] += 1
            if is_train:
                close(summary["train_steps"], sum(r["steps"] for r in rows), str(folder))
                counts["raw_to_summary_checks"] += 1
        csv_row = csv_summary[(summary["config"], summary["algorithm"], summary["seed"])]
        assert set(csv_row) == set(summary)
        for key, value in summary.items():
            if key in ("config", "algorithm"):
                assert csv_row[key] == value
            else:
                close(csv_row[key], value, (str(folder), key))
                counts["summary_csv_checks"] += 1
    groups = json.loads((root / "aggregate.json").read_text())
    with (root / "aggregate.csv").open(encoding="utf-8-sig") as stream:
        csv_groups = {(r["config"], r["algorithm"]): r for r in csv.DictReader(stream)}
    assert len(csv_groups) == len(groups) == len(metadata["configurations"]) * 2
    for group in groups:
        values = [json.loads(f.read_text()) for f in
                  (root/group["config"]/group["algorithm"]).glob("seed_*/summary.json")]
        assert len(values) == group["n_seeds"]
        for key in values[0]:
            if key in ("config", "algorithm", "seed"):
                continue
            nums = [v[key] for v in values]
            assert abs(statistics.mean(nums)-group[key+"_mean"]) < 1e-10
            sd = statistics.stdev(nums) if len(nums) > 1 else 0
            assert abs(sd-group[key+"_sd"]) < 1e-10
            counts["aggregate_checks"] += 2
        csv_group = csv_groups[(group["config"], group["algorithm"])]
        assert set(csv_group) == set(group)
        for key, value in group.items():
            if key not in ("config", "algorithm"):
                close(csv_group[key], value, (group["config"], group["algorithm"], key))
                counts["aggregate_checks"] += 1
    print(json.dumps(dict(status="passed", **counts), indent=2))


if __name__ == "__main__":
    main()
