"""仅从保存的 CSV / JSON / NPY 生成图，不在绘图时重新训练。"""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

COLORS = {"q_learning": "#B44B35", "sarsa": "#247B9B"}
LABELS = {"q_learning": "Q-learning", "sarsa": "Sarsa"}


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def finish(fig, path):
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def grid(ax):
    for row in range(4):
        for col in range(12):
            color = "#F8FAFC"
            if row == 3 and 1 <= col <= 10:
                color = "#5C6673"
            elif (row, col) == (3, 0):
                color = "#E0EDF5"
            elif (row, col) == (3, 11):
                color = "#DCEEDB"
            ax.add_patch(Rectangle((col, row), 1, 1, facecolor=color,
                                   edgecolor="#B4BDC7", linewidth=0.7))
    ax.text(.5, 3.5, "S", ha="center", va="center", weight="bold", fontsize=12)
    ax.text(11.5, 3.5, "G", ha="center", va="center", weight="bold", fontsize=12)
    ax.text(6, 3.5, "CLIFF", ha="center", va="center", color="white")
    ax.set(xlim=(0, 12), ylim=(4, 0), aspect="equal")
    ax.set_xticks(np.arange(12) + .5, np.arange(12))
    ax.set_yticks(np.arange(4) + .5, np.arange(4))
    ax.tick_params(length=0, labelsize=8)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    for spine in ax.spines.values():
        spine.set_visible(False)


def create_figures(results):
    results = Path(results)
    out = results / "figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18})
    data = json.loads((results / "aggregate.json").read_text(encoding="utf-8"))
    lookup = {(d["config"], d["algorithm"]): d for d in data}
    fig, ax = plt.subplots(figsize=(10, 3.5), layout="constrained")
    grid(ax)
    ax.set_title("Cliff Walking   |   4 x 12 grid   |   shortest safe route: 13 steps")
    finish(fig, out / "environment.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for algorithm in LABELS:
        files = sorted((results / "baseline" / algorithm).glob("seed_*/training.csv"))
        all_rows = [load_csv(path) for path in files]
        window = min(50, len(all_rows[0]))
        for ax, metric, title in zip(axes, ("return_", "falls"),
                                     ("Training return", "Cliff falls per episode")):
            values = np.array([[float(r[metric]) for r in rows] for rows in all_rows])
            smooth = np.array([np.convolve(v, np.ones(window)/window, mode="valid") for v in values])
            x = np.arange(window, values.shape[1] + 1)
            mean, sd = smooth.mean(axis=0), smooth.std(axis=0, ddof=1 if len(files) > 1 else 0)
            ax.plot(x, mean, color=COLORS[algorithm], label=LABELS[algorithm], linewidth=1.5)
            ax.fill_between(x, mean-sd, mean+sd, color=COLORS[algorithm], alpha=.13)
            ax.set(xlabel="Episode", ylabel=title)
    axes[0].set_yscale("symlog", linthresh=50)
    axes[1].set_yscale("symlog", linthresh=.5)
    axes[0].legend()
    fig.suptitle(f"Baseline: {window}-episode moving mean; shaded area = across-seed SD")
    finish(fig, out / "baseline_learning.png")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout="constrained")
    for ax, algorithm in zip(axes, LABELS):
        grid(ax)
        route = json.loads((results / "baseline" / algorithm / "seed_00" / "route.json").read_text())
        xy = np.array([[s % 12 + .5, s // 12 + .5] for s in route["states"]])
        ax.plot(xy[:, 0], xy[:, 1], color=COLORS[algorithm], linewidth=2.8, marker="o", markersize=3)
        for start, end in zip(xy[:-1], xy[1:]):
            ax.annotate("", xy=end, xytext=start,
                        arrowprops=dict(arrowstyle="->", color=COLORS[algorithm], lw=1.6))
        ax.set_title(f"{LABELS[algorithm]} | preselected seed 0 | "
                     f"{len(route['actions'])} steps | {route['reason']}")
    finish(fig, out / "baseline_routes.png")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), layout="constrained")
    for ax, metric, title in zip(axes, ("return_", "steps", "falls"),
                                 ("Episode return", "Steps per episode", "Cliff falls per episode")):
        for i, algorithm in enumerate(LABELS):
            d = lookup[("baseline", algorithm)]
            values = [d[f"{mode}_{metric}_mean"] for mode in ("greedy", "exploratory")]
            errors = [d[f"{mode}_{metric}_sd"] for mode in ("greedy", "exploratory")]
            ax.bar(np.arange(2)+(i-.5)*.32, values, .32, yerr=errors, capsize=3,
                   color=COLORS[algorithm], label=LABELS[algorithm])
        ax.set_xticks([0, 1], ["Greedy", "Explore 0.1"])
        ax.set_ylabel(title)
    axes[1].legend(fontsize=8)
    fig.suptitle("Frozen final policies | mean +/- SD of seed-level means")
    finish(fig, out / "baseline_evaluation.png")

    groups = [
        ("gamma", ["gamma_0", "gamma_0.5", "gamma_0.9", "baseline"],
         ["0", "0.5", "0.9", "1"], "Discount factor"),
        ("cliff", ["cliff_m1", "cliff_m10", "baseline", "cliff_m500", "cliff_m1000"],
         ["-1", "-10", "-100", "-500", "-1000"], "Cliff reward"),
        ("step", ["step_m0.1", "baseline", "step_m2", "step_m5"],
         ["-0.1", "-1", "-2", "-5"], "Step reward"),
        ("epsilon", ["epsilon_0", "baseline", "epsilon_0.3"],
         ["0", "0.1", "0.3"], "Training epsilon"),
    ]
    for name, configs, labels, xlabel in groups:
        if not all((config, "sarsa") in lookup for config in configs):
            continue
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
        metrics = ("standard_return", "falls", "steps", "success")
        titles = ("Common score (-steps - 99 x falls)", "Cliff falls per episode",
                  "Steps per episode (including timeouts)", "Goal-reaching rate")
        for ax, metric, title in zip(axes.flat, metrics, titles):
            for algorithm in LABELS:
                values = [lookup[(c, algorithm)][f"exploratory_{metric}_mean"] for c in configs]
                errors = [lookup[(c, algorithm)][f"exploratory_{metric}_sd"] for c in configs]
                ax.errorbar(range(len(configs)), values, yerr=errors, color=COLORS[algorithm],
                            marker="o", capsize=3, label=LABELS[algorithm], linewidth=1.6)
            ax.set_xticks(range(len(configs)), labels)
            ax.set(xlabel=xlabel, ylabel=title)
            if metric == "success":
                ax.set_ylim(-.03, 1.07)
            elif name == "gamma" and metric == "standard_return":
                ax.set_yscale("symlog", linthresh=50)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("Frozen-policy evaluation with common epsilon = 0.1 | seed mean +/- SD")
        finish(fig, out / f"sensitivity_{name}.png")
    # γ 对训练曲线的作用，保留低 γ 的长回合及截断数据。
    gamma_names = ["gamma_0", "gamma_0.5", "gamma_0.9", "baseline"]
    if all((name, "sarsa") in lookup for name in gamma_names):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
        for ax, name, gamma in zip(axes.flat, gamma_names, (0, .5, .9, 1)):
            for algorithm in LABELS:
                rows = [load_csv(p) for p in sorted((results/name/algorithm).glob("seed_*/training.csv"))]
                values = np.array([[float(r["return_"]) for r in run] for run in rows])
                window = min(50, values.shape[1])
                sm = np.array([np.convolve(v, np.ones(window)/window, "valid") for v in values])
                ax.plot(np.arange(window, values.shape[1]+1), sm.mean(0),
                        color=COLORS[algorithm], label=LABELS[algorithm])
            ax.set(title=f"gamma = {gamma}", xlabel="Episode", ylabel="Training return")
            ax.set_yscale("symlog", linthresh=50)
        axes[0, 0].legend(fontsize=8)
        finish(fig, out / "gamma_learning.png")
    # 奖励敏感性也展示训练过程；每一子图在同一奖励下比较算法。
    # Main panel matches the four cliff penalties explicitly named in class.
    # The extra -500 experiment remains in the final-policy comparison and tables.
    cliff_names = ["cliff_m1", "cliff_m10", "baseline", "cliff_m1000"]
    if all((name, "sarsa") in lookup for name in cliff_names):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
        for ax, name, value in zip(axes.flat, cliff_names, (-1, -10, -100, -1000)):
            for algorithm in LABELS:
                runs = [load_csv(p) for p in sorted((results/name/algorithm).glob("seed_*/training.csv"))]
                values = np.array([[float(r["return_"]) for r in run] for run in runs])
                window = min(50, values.shape[1])
                sm = np.array([np.convolve(v, np.ones(window)/window, "valid") for v in values])
                ax.plot(np.arange(window, values.shape[1]+1), sm.mean(0),
                        color=COLORS[algorithm], label=LABELS[algorithm], linewidth=1.2)
            ax.set(title=f"Cliff reward = {value}", xlabel="Episode", ylabel="Raw training return")
            ax.set_yscale("symlog", linthresh=25)
        axes[0, 0].legend(fontsize=8)
        finish(fig, out / "cliff_learning.png")
    step_names = ["step_m0.1", "baseline", "step_m2", "step_m5"]
    if all((name, "sarsa") in lookup for name in step_names):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for ax, algorithm in zip(axes, LABELS):
            for name, value, color in zip(step_names, (-.1, -1, -2, -5),
                                         ("#8064A2","#247B9B","#39835B","#B44B35")):
                runs = [load_csv(p) for p in sorted((results/name/algorithm).glob("seed_*/training.csv"))]
                values = np.array([[float(r["return_"]) for r in run] for run in runs])
                window = min(50, values.shape[1])
                sm = np.array([np.convolve(v, np.ones(window)/window, "valid") for v in values])
                ax.plot(np.arange(window, values.shape[1]+1), sm.mean(0),
                        color=color, label=f"Step reward = {value}", linewidth=1.2)
            ax.set(title=LABELS[algorithm], xlabel="Episode", ylabel="Raw training return")
            ax.set_yscale("symlog", linthresh=10)
            ax.legend(fontsize=8)
        fig.suptitle("Reward scales differ: use steps and falls for cross-setting comparisons")
        finish(fig, out / "step_learning.png")
    print(f"Figures saved to {out.resolve()}", flush=True)
