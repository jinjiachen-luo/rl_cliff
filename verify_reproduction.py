"""重放全部训练和评估，逐项核对原始日志、Q 表与路线；不覆盖实验数据。

python verify_reproduction.py --results results --workers 4
计时受设备与调度影响，不作为可精确复现的数值核验。
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import time

import numpy as np
from environment import Config
from experiment import evaluate, greedy_route, train


def check_records(path, actual):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        saved = list(csv.DictReader(stream))
    assert len(actual) == len(saved), str(path)
    for i, (a, b) in enumerate(zip(actual, saved), 1):
        assert set(a) == set(b), (str(path), i)
        for key in a:
            assert float(a[key]) == float(b[key]), (str(path), i, key, a[key], b[key])
    return len(saved)


def verify_model(folder):
    folder = Path(folder)
    config = Config(**json.loads((folder / "config.json").read_text()))
    algorithm = folder.parent.name
    seed = int(folder.name.removeprefix("seed_"))
    q, records, _ = train(config, algorithm, seed)
    training_rows = check_records(folder / "training.csv", records)
    assert np.array_equal(np.asarray(q), np.load(folder / "q_table.npy")), str(folder)
    evaluation_rows = 0
    for mode, epsilon, offset in (("greedy", 0.0, 0), ("exploratory", 0.1, 10000)):
        evaluated = evaluate(q, config, epsilon, 100000 + seed + offset)
        evaluation_rows += check_records(folder / f"eval_{mode}.csv", evaluated)
    saved_route = json.loads((folder / "route.json").read_text())
    assert greedy_route(q, config) == saved_route, str(folder)
    return training_rows, evaluation_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = Path(args.results)
    folders = sorted(root.glob("*/*/seed_*"))
    metadata = json.loads((root / "metadata.json").read_text())
    expected = len(metadata["configurations"]) * len(metadata["seeds"]) * 2
    assert len(folders) == expected
    started = time.perf_counter()
    training_rows = evaluation_rows = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(verify_model, str(folder)) for folder in folders]
        for completed, future in enumerate(as_completed(pending), 1):
            n_train, n_eval = future.result()
            training_rows += n_train
            evaluation_rows += n_eval
            if completed % 20 == 0 or completed == len(folders):
                print(f"Verified {completed}/{len(folders)} models", flush=True)
    result = {
        "status": "passed",
        "models_retrained_and_matched": len(folders),
        "q_tables_exactly_matched": len(folders),
        "routes_exactly_matched": len(folders),
        "training_rows_exactly_matched": training_rows,
        "evaluation_rows_exactly_matched": evaluation_rows,
        "wall_seconds": time.perf_counter() - started,
        "timing_values_compared": False,
        "original_data_overwritten": False,
    }
    (root / "reproduction_check.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
