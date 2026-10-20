"""从既有实验日志补充逐算法曲线、阶段统计与风险分析，不重新训练。

运行：python detailed_analysis.py --results results
"""
import argparse
import csv
import json
from pathlib import Path
import statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator, FuncFormatter

ALGORITHMS = {"q_learning": ("Q-learning", "#B44B35"),
              "sarsa": ("Sarsa", "#247B9B")}
STAGES = [(1, 100), (101, 500), (501, 1500), (1501, 3000)]


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [{k: float(v) for k, v in row.items()}
                for row in csv.DictReader(stream)]


def summarize(values):
    return {"mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0}


def numerical_ticks(ax):
    """给对称对数轴补充可直接读数的刻度，而非只显示一个 -100。"""
    low, high = ax.get_ylim()
    candidates = [-10000, -5000, -2000, -1000, -500, -200, -100,
                  -50, -25, -10, -5, -2, -1, 0]
    ticks = [v for v in candidates if low <= v <= high]
    if len(ticks) > 7:
        ticks = ticks[::2]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    root = Path(args.results)
    output = root / "detailed"
    output.mkdir(exist_ok=True)
    chinese_font = Path("C:/Windows/Fonts/simhei.ttf")
    if chinese_font.exists():
        font_manager.fontManager.addfont(str(chinese_font))
        family = font_manager.FontProperties(fname=str(chinese_font)).get_name()
    else:
        family = "sans-serif"
    plt.rcParams.update({"font.family": family, "font.size": 11,
                         "axes.unicode_minus": False, "axes.grid": True,
                         "grid.alpha": .2, "axes.spines.top": False,
                         "axes.spines.right": False})
    data = {}
    for algorithm, (label, color) in ALGORITHMS.items():
        folders = sorted((root/"baseline"/algorithm).glob("seed_*"))
        runs = [rows(folder/"training.csv") for folder in folders]
        stage_stats = []
        for first, last in STAGES:
            runstats = []
            for run in runs:
                block = run[first-1:last]
                if not block:
                    continue
                runstats.append({
                    "return": statistics.mean(r["return_"] for r in block),
                    "falls": statistics.mean(r["falls"] for r in block),
                    "steps": statistics.mean(r["steps"] for r in block),
                    "fall_episode_rate": statistics.mean(r["falls"] > 0 for r in block),
                    "success": statistics.mean(r["success"] for r in block),
                })
            if runstats:
                stage_stats.append({"first": first, "last": last,
                                    **{key: summarize([r[key] for r in runstats])
                                       for key in runstats[0]}})
        late_stats = []
        for run in runs:
            block = run[-1000:]
            late_stats.append({
                "return": statistics.mean(r["return_"] for r in block),
                "within_run_return_sd": statistics.stdev(r["return_"] for r in block),
                "fall_episode_rate": statistics.mean(r["falls"] > 0 for r in block),
                "falls": statistics.mean(r["falls"] for r in block),
                "steps": statistics.mean(r["steps"] for r in block),
            })
        evalstats = []
        for folder in folders:
            evaluated = rows(folder/"eval_exploratory.csv")
            evalstats.append({
                "fall_episode_rate": statistics.mean(r["falls"] > 0 for r in evaluated),
                "falls_per100": statistics.mean(r["falls"] for r in evaluated)*100,
                "edge_fraction": sum(r["edge_steps"] for r in evaluated)/sum(r["steps"] for r in evaluated),
                "step_cost": statistics.mean(r["steps"] for r in evaluated),
                "extra_cliff_cost": 99*statistics.mean(r["falls"] for r in evaluated),
            })
        seed0 = runs[0]
        reward = np.array([r["return_"] for r in seed0])
        smooth = np.convolve(reward, np.ones(50)/50, "valid")
        fig, axes = plt.subplots(2, 1, figsize=(10, 6.6), layout="constrained")
        for ax, first in zip(axes, (1, max(1,len(seed0)-999))):
            x = np.arange(first, len(seed0)+1)
            ax.plot(x, reward[first-1:], color="#9AA0A6", alpha=.65,
                    linewidth=.65, label="种子 0 逐回合原始回报")
            start = max(first,50)
            ax.plot(np.arange(start,len(seed0)+1), smooth[start-50:],
                    color=color, linewidth=1.6, label="50 回合移动均值")
            ax.set(xlabel="训练回合", ylabel="未折扣回报",
                   title="完整训练过程" if first==1 else "最后 1000 回合局部放大")
            ax.set_yscale("symlog", linthresh=25)
            numerical_ticks(ax)
            ax.legend(loc="lower right", fontsize=9)
        fig.suptitle(f"{label} 单独训练曲线    基准参数    预先指定种子 0")
        fig.savefig(output/f"{algorithm}_individual.png",dpi=220,bbox_inches="tight")
        plt.close(fig)
        data[algorithm] = {
            "stages": stage_stats,
            "last1000": {key:summarize([r[key] for r in late_stats]) for key in late_stats[0]},
            "evaluation": {key:summarize([r[key] for r in evalstats]) for key in evalstats[0]},
            "seed0": {
                "first100_return": statistics.mean(r["return_"] for r in seed0[:100]),
                "last1000_return": statistics.mean(r["return_"] for r in seed0[-1000:]),
                "last1000_fall_episodes": sum(r["falls"]>0 for r in seed0[-1000:]),
                "worst_return": min(r["return_"] for r in seed0),
            },
        }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
    for algorithm, (label,color) in ALGORITHMS.items():
        evaluated=data[algorithm]["evaluation"]
        index=list(ALGORITHMS).index(algorithm)
        axes[0].bar(index,evaluated["step_cost"]["mean"],color="#AAB5C4",
                    label="步数代价" if index==0 else None)
        axes[0].bar(index,evaluated["extra_cliff_cost"]["mean"],
                    bottom=evaluated["step_cost"]["mean"],color=color,
                    label=f"{label} 坠崖额外代价")
    axes[0].set_xticks([0,1],["Q-learning","Sarsa"])
    axes[0].set(ylabel="平均损失（越小越好）",title="回报损失分解：-R = T + 99F")
    axes[0].legend(fontsize=8)
    for i,(algorithm,(label,color)) in enumerate(ALGORITHMS.items()):
        ev=data[algorithm]["evaluation"]
        values=[100*ev["fall_episode_rate"]["mean"],100*ev["edge_fraction"]["mean"]]
        errors=[100*ev["fall_episode_rate"]["sd"],100*ev["edge_fraction"]["sd"]]
        axes[1].bar(np.arange(2)+(i-.5)*.34, values,.34,yerr=errors,
                    color=color,capsize=3,label=label)
    axes[1].set_xticks([0,1],["有坠崖的回合比例","贴崖状态访问比例"])
    axes[1].set(ylabel="比例 / %",title="统一探索评估下的风险")
    axes[1].legend(fontsize=9)
    fig.savefig(output/"risk_decomposition.png",dpi=220,bbox_inches="tight")
    plt.close(fig)
    (output/"analysis.json").write_text(json.dumps(data,ensure_ascii=False,indent=2),
                                       encoding="utf-8")
    print(json.dumps(data,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
