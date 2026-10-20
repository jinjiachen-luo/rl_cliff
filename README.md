# Cliff Walking 悬崖漫步实验

从零实现表格 Q-learning 与一步 Sarsa。环境为 4×12 确定性网格，
起点 `(3,0)`，终点 `(3,11)`，悬崖为 `(3,1)` 到 `(3,10)`。
悬崖奖励替代普通步长奖励，掉崖后回到起点但不结束回合；到达终点才终止。

## 运行

已在 Windows、Python 3.10.22、NumPy 2.2.6、Matplotlib 3.10.9 下运行验证。
无需 GPU、Gymnasium 或 PyTorch。若使用现有 `rlcourse` 环境：

```powershell
conda activate rlcourse
cd "D:\Users\DELL\Desktop\大四上\强化学习\第二次上机\cliff_experiment"
python -m unittest -v
python main.py run --output results_new --workers 4
```

在其他 Python 3.10+ 环境中，可先执行：

```powershell
python -m pip install -r requirements.txt
```

`results` 是本次报告使用的真实结果。复现实验建议输出到 `results_new`，
避免覆盖报告对应的数据。完整实验包含 13 组配置 × 2 种算法 × 10 个种子，
每次训练 3000 回合，最多 1000 步；每个最终模型分别进行 200 回合贪心评估
和 200 回合 epsilon=0.1 评估。计算耗时会随设备变化。

只运行基准对照：

```powershell
python main.py run --baseline-only --output baseline_new --workers 4
```

从现有数据重新绘图或载入一个模型：

```powershell
python main.py plot --results results
python detailed_analysis.py --results results
python main.py evaluate results/baseline/sarsa/seed_00 --epsilon 0.1
python audit_results.py --results results
python verify_reproduction.py --results results --workers 4
```

如果多进程受到环境限制，应在本地终端执行；Windows 多进程入口已使用
`if __name__ == "__main__"` 保护。

## 文件说明

| 文件 | 内容 |
|---|---|
| `environment.py` | 环境转移、奖励和配置 |
| `algorithms.py` | epsilon-greedy 与两种 TD 更新 |
| `experiment.py` | 训练、冻结评估、汇总和存储 |
| `plotting.py` | 训练曲线、路线、参数对照图 |
| `detailed_analysis.py` | 分别绘制两种算法的原始曲线，统计训练阶段与风险分解 |
| `main.py` | 命令行入口 |
| `test_core.py` | 10 项关键语义测试 |
| `audit_results.py` | 数据完整性与统计复核 |
| `verify_reproduction.py` | 重放所有训练和评估，逐项比对原始记录、Q 表及路线 |
| `results/aggregate.csv`、`aggregate.json` | 跨种子均值与样本标准差 |
| `results/seed_summary.csv` | 每个配置、算法和训练种子的统计量 |
| `results/metadata.json` | 参数、版本、随机数规则及实验源码哈希 |
| `results/figures/` | 报告插图 |
| `results/detailed/` | 各算法原始曲线、后期放大图、阶段统计及风险分解 |
| `results/<配置>/<算法>/seed_XX/` | Q 表、逐回合日志、评估日志、路线和配置 |

原始 CSV 字段 `return_` 表示未折扣累计环境奖励；`standard_return` 表示
统一重评分 `-steps - 99*falls`；`success` 与 `truncated` 分别表示到达终点和
超时。悬崖掉落不是 `terminated`，截断也不把 TD 目标中的下一状态价值清零。

## 实验协议和阅读结果时的注意事项

- 基准：alpha=0.1、gamma=1、epsilon=0.1、普通奖励=-1、悬崖奖励=-100。
- 单因素：gamma={0,0.5,0.9,1}；悬崖奖励={-1,-10,-100,-500,-1000}；
  普通奖励={-0.1,-1,-2,-5}；训练 epsilon={0,0.1,0.3}。重复基准只运行一次。
- 按课堂口述补充的悬崖奖励 -1000 和普通奖励 -2 各新增 20 个模型。
  原有 220 个模型文件保持不变；完整实验共 260 个模型。
  `metadata.json` 保存补充批次信息及初始、当前源码哈希。
  `run.log` 保留最初 220 个模型的运行日志；新增批次见
  `metadata.json` 的 `classroom_reward_extension` 及新增模型目录。
- 种子为 0–9。训练与评估随机流独立；两种算法的同编号种子不代表相同轨迹。
- 先求每个模型 200 回合的均值，再求 10 个训练种子的均值和样本标准差。
  图中阴影/误差棒是跨种子标准差，不是 95% 置信区间。
- 无论训练 epsilon 是多少，统一安全性对照都在 epsilon=0.1 下评估。
  epsilon=0 评估仍对完全相等的最大 Q 值随机打破并列。
- 路线图固定展示预先指定的种子 0，以最小动作编号打破并列，仅用于画图。
  路线图不能替代 10 个模型的评估统计。
- gamma=0 时不学习长期目标；gamma=0.5 的 Sarsa 及 epsilon=0.3 的个别模型
  有超时/循环，全部保留在统计中。成功率必须与坠崖次数同时看。
- 相同训练回合数不等于相同环境交互步数，实际步数单独保存；
  本实验没有据此声称某种算法样本效率更高。
- 原始奖励不同的实验不能只比较 `return_` 大小，应结合步数、坠崖、到达率
  和统一重评分。
- 固定学习率、固定探索率、有限训练次数的结果不构成一般收敛保证。
- 补充分析直接读取同一批实验日志，不重新训练或筛选种子。
  原始曲线展示预先指定种子 0，阶段统计和风险比较使用全部 10 个种子。
  `fall_episode_rate` 是至少坠崖一次的回合比例；`falls_per100`
  是每 100 回合的坠崖次数，两者不相等，因为一个回合可能多次坠崖。
- `verify_reproduction.py` 只在内存中重新训练和评估，不覆盖原始日志和 Q 表；
  核验结果保存为 `results/reproduction_check.json`。实测耗时受机器与调度影响，
  不参与精确一致性比较。Windows 多进程核验需允许本机进程间通信。

## 参考

1. 本次提供的《第二次上机.docx》。
2. Sutton 与 Barto，Reinforcement Learning: An Introduction，第 6 章、Cliff Walking 例。
   <https://web.stanford.edu/class/psych209/Readings/SuttonBartoRL.pdf>
3. Farama Foundation，Cliff Walking 官方环境说明。
   <https://gymnasium.farama.org/environments/toy_text/cliff_walking/>
4. 用户提供的实现参考：RL 实践（3）—— 悬崖漫步。
   <https://blog.csdn.net/wxc971231/article/details/128180067>

参考网页包含其他动作集合、终止和终点奖励设置的变体，本程序采用上述明确规则。
没有移植其 replay buffer 或额外算法。用户提供的知乎页面
<https://zhuanlan.zhihu.com/p/15572517925> 在本次访问中未能读取，
没有引用其具体内容。所有报告数值均来自随附 results 数据。
