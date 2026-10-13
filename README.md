# Lending Club 贷款违约预测

一个面向**数据分析 / 风控建模**岗位的作品集项目。基于 Kaggle 上
`emmaruyiyang/lending-club-loan-default-prediction-eda` 的骨架重写，
但修正了原版 4 个会让结果失真的问题，并补充了业务分析部分。

---

## 零、怎么运行（专用虚拟环境）

这个项目使用一个**独立的虚拟环境**，不依赖你系统的全局 Python。

```bash
# 运行方式（注意路径）
"D:/venvs/lending-club/Scripts/python.exe" pipeline.py --smoke
```

在 VS Code 里：`Ctrl+Shift+P` → `Python: Select Interpreter` →
选择 `D:\venvs\lending-club\Scripts\python.exe`，之后直接按运行按钮即可。

三种运行模式：

| 命令 | 作用 |
|------|------|
| `pipeline.py --smoke` | 用合成数据跑通全流程，**不需要下载数据集**，先跑这个 |
| `pipeline.py` | 修正版：真实数据，无泄漏 |
| `pipeline.py --leaky` | 原版口径：故意保留泄漏字段（对比实验用，**只对真实数据有差别**） |
| `pipeline.py --balance undersample` | 复现原版的欠采样做法，用来对比 |

> 注：`--smoke` 模式下合成数据里没有泄漏字段，所以 `--leaky` 的结果和修正版一样。
> 想看真实的泄漏差距，必须用真实数据集。

---

## 一、这个项目要回答的业务问题

> **在放款前的时点上，用申请人当时就能拿到的信息，预测这笔贷款会不会违约。**

这句话里有两个关键词，是整个项目的灵魂，也是面试必问：

1. **"放款前"** —— 所以任何放款之后才产生的字段（最后一次还款金额、剩余本金、
   最后一次征信查询时间、`last_fico_range_*` 等）**都不能用**。用了就是**时间泄漏**，
   模型在离线测试上分数很漂亮，上线后直接崩。
2. **"会不会违约"** —— 这是一个**业务定义**，不是数据里现成的。`Late (16-30 days)`
   算不算违约？`In Grace Period` 呢？这个选择直接决定坏账率是多少，
   也决定了模型上线后要抓的是什么人。

---

## 二、数据获取

推荐数据集：`wordsforthewise/lending-club` 中的 `accepted_2007_to_2018Q4.csv`（约 1.7 GB，
220 万行 × 151 列）。

**方式 A：Kaggle CLI（推荐，一次配好长期可用）**

```bash
pip install kaggle
# 1. 注册/登录 kaggle.com  ->  右上头像 -> Settings -> API -> Create New Token
# 2. 下载到的 kaggle.json 放到  C:\Users\小芬\.kaggle\kaggle.json
kaggle datasets download -d wordsforthewise/lending-club -p D:/data/lending-club --unzip
```

**方式 B：网页手动下载**
文件太大，注意下载到 D 盘（C 盘只剩 39G）。

> ⚠️ **内存警告**：你的机器 15.7G 内存。**不要**直接 `pd.read_csv(整个文件)`，
> 会爆。`pipeline.py` 里已经写好了 `usecols=需要的列` + 抽样 10 万行，
> 峰值内存可控在 2G 以内。

### 实际位置（已就位）

```
C:/Users/小芬/lending-club-project/data/accepted_2007_to_2018Q4.csv.gz   (374 MB)
```

**为什么是 `.gz` 而不是 `.csv`**：压缩包只有 374 MB，解压后是 1.7 GB。
pandas 的 `read_csv` 能自动识别并解压 `.gz`，**不需要手动解压**，
也省了 1.3 GB 磁盘。读全量 226 万行的耗时约 142 秒。

`pipeline.py` 会按下面的顺序自动查找，找到就用；也可以用 `--data 路径` 显式指定：

```
C:/Users/小芬/lending-club-project/data/accepted_2007_to_2018Q4.csv.gz
C:/Users/小芬/lending-club-project/data/accepted_2007_to_2018Q4.csv
D:/data/lending-club/accepted_2007_to_2018Q4.csv.gz
D:/data/lending-club/accepted_2007_to_2018Q4.csv
```

**数据集里的标签分布**（本次实测，抽样 10 万行）：

| loan_status | 条数 | 归类 |
|---|---|---|
| Fully Paid | 47,460 | ✅ 好账 |
| Charged Off | 12,034 | ❌ 坏账 |
| Current | 38,887 | 🗑️ 结局未定，剔除 |
| Late (31-120 days) | 969 | ❌ 坏账 |
| In Grace Period | 368 | 🗑️ 结局未定，剔除 |
| Late (16-30 days) | 187 | 🗑️ 结局未定，剔除 |
| Does not meet the credit policy（两种） | 91 | 按 Fully Paid / Charged Off 分别归类 |

注意 **`Current` 占 39%**，全部要剔掉——这也正是原版 notebook 做得对的地方。
剔除后坏账率约 **21.5%**，这个数字就是全文所有指标的基准线参照。

---

## 三、7 天计划

| Day | 任务 | 产出 |
|-----|------|------|
| **1** | 装环境、下数据、`python pipeline.py --smoke` 跑通 | 环境就绪 |
| **2** | `python pipeline.py` 跑真实数据；读 EDA 部分，产出**业务问题定义 + 坏账率拆解** | 3-5 张 EDA 图 |
| **3** | 逐节读清洗和特征工程代码，**每一步都问自己"为什么"** | 一份自己的笔记 |
| **4** | ⭐ **做「泄漏修正前后对比实验」** | 关键结论表 |
| **5** | 补 DA 侧内容：同一套指标用 SQL 写一遍 + 可视化 | `analysis.sql` |
| **6** | 写报告（就是这份 README 的扩写版） | 完整 README |
| **7** | 简历包装 + 模拟面试追问 | 简历条目 |

### ⭐ Day 4 是重点：修正前后对比实验

这是**整个项目最值钱的部分**，也是你和别人拉开差距的地方。做法：

1. 跑一遍原版（保留泄漏字段），记录 AUC / KS；
2. 跑一遍修正版（删掉泄漏字段），记录 AUC / KS；
3. 把两个数字并排放，**解释差值从哪来**。

面试时你可以这样讲：

> "我复现这个 Notebook 时发现它的 AUC 有 0.91，但我一看特征列表就觉得不对——
> 里面 `last_fico_range_high/low` 与标签的相关性高达 -0.66 和 -0.56，
> 排在所有特征前两位。这两个字段是贷款存续期间每月更新的 FICO 分，
> 放款时根本拿不到。把这类字段删掉后 AUC 掉到了 X.XX，这个才是这个模型真实的预测能力。"

**这段话的价值远超模型本身**——它证明你不是"调包侠"，而是真的理解业务。

---

## 四、原版的问题清单（也是你的面试素材）

| # | 问题 | 证据 | 修复 |
|---|------|------|------|
| 1 | **标签泄漏**：`loan_condition` / `loan_status` 没从特征里删 | 实测 `loan_condition_Good Loan` 与标签相关系数 = **-1.0** | 统一进 `LEAK_COLS`，建模前加断言 |
| 2 | **时间泄漏**：用了 `last_fico_range_*`、`last_pymnt_d`、`last_credit_pull_d` | 原版相关性最高的两个特征就是 `last_fico_range_*` | 全部进 `POST_ORIGINATION` |
| 3 | **特征选择失效**：对标准化数据做 `VarianceThreshold(threshold=1)` | 标准化后方差恒为 1，`var>1` 等价抛硬币（实测 40 个随机留 19 个） | 删掉；改用折内互信息 / 模型重要性 |
| 4 | **Stacking 用样本内预测训练元模型** | `fit_first_level` 里 `fit(X_trn)` 后 `predict(X_trn)` | 改用 out-of-fold 预测 |
| 5 | CV 用 `KFold` 而非 `StratifiedKFold`，且无 `random_state` | — | `StratifiedKFold(shuffle=True, random_state=42)` |
| 6 | **先欠采样再做 CV**，导致 CV 在 1:1 平衡集上评估 | 真实坏账率 21.9%，CV 里变成 50% | 重采样移进每一折的训练部分 |
| 7 | 标签定义把 `Late (16-30 days)` / `In Grace Period` 也算坏账 | — | 明确区分「确定坏 / 确定好 / 模糊剔除」 |
| 8 | `fillna(0)` 兜底，`mths_since_last_delinq` 缺失=从未逾期，填 0 语义相反 | — | 按语义填充 + 缺失指示变量 |

---

## 五、项目结构

```
lending-club-project/
├── README.md            # 本文件（项目说明）
├── 简历与面经.md         # 简历条目（DA 版 / 建模版）+ 高频面试题与答法
├── requirements.txt
├── pipeline.py          # 建模流水线（主交付物），支持 --leaky / --split temporal
├── eda.py               # EDA 可视化，输出 8 张业务图
├── analysis.sql         # 核心指标的 SQL 实现（8 组查询）
├── run_sql.py           # 执行 analysis.sql（DuckDB），结果存 sql_results.txt
├── cost_threshold.py    # 成本敏感阈值：不取 0.5，取代价最小的审批线
├── scorecard.py         # WOE + IV 评分卡（分箱 → 粗分类 → 去冗余 → LR → 评分表）
├── clustering.py        # 借款人聚类分群（无监督，轮廓系数 vs 业务指标选 k）
├── ab_test.py           # A/B 实验设计与显著性检验 + 统计功效分析
├── psm.py               # 倾向得分匹配：期限对违约率的因果效应
├── figures/             # 15 张图
│   ├── 01_target_definition.png     标签口径与坏账率基线
│   ├── 02_bad_rate_by_grade.png     评级 A→G 单调性验证
│   ├── 03_bad_rate_by_fico_intrate.png
│   ├── 04_bad_rate_by_term_purpose.png
│   ├── 05_bad_rate_by_region_year.png
│   ├── 06_bad_rate_by_dti_income.png
│   ├── 07_corr_with_target.png      ★ 时间泄漏的证据
│   ├── 08_missing_top.png           缺失语义
│   ├── 09_cost_threshold.png        ★ 成本曲线与最优阈值
│   ├── 10_scorecard_iv.png          特征 IV 排序
│   ├── 11_scorecard_score.png       ★ 评分十分位坏账率（单调性验证）
│   ├── 12_cluster_selection.png     ★ 轮廓系数会选错 k
│   ├── 13_cluster_profile.png       ★ 分群画像与坏账率
│   ├── 14_ab_test.png               显著性检验与功效曲线
│   └── 15_psm.png                   ★ 倾向得分重叠 + Love plot
├── run_fixed.log        # 随机划分运行日志
├── run_leaky.log        # 含泄漏版运行日志
├── run_temporal.log     # 时间外验证运行日志
├── sql_results.txt      # SQL 全部查询结果（可复核）
├── cost_analysis.txt    # 成本阈值完整结果
├── scorecard.txt        # 评分卡完整表（含 WOE 明细）
├── clustering.txt       # 分群画像与选 k 依据
├── ab_test.txt          # 检验与功效分析完整结果
├── psm.txt              # 匹配平衡性与 ATT 估计
└── data/                # 数据集与缓存（不提交）
```

### 一键复现

```bash
PY="D:/venvs/lending-club/Scripts/python.exe"
"$PY" pipeline.py                    # 修正版建模 → run_fixed.log
"$PY" pipeline.py --leaky            # 含泄漏版 → run_leaky.log
"$PY" pipeline.py --split temporal   # 时间外验证 → run_temporal.log
"$PY" eda.py                         # 生成 figures/01-08
"$PY" run_sql.py                     # 跑 analysis.sql → sql_results.txt
"$PY" cost_threshold.py              # 成本敏感阈值 → cost_analysis.txt + 图 9
"$PY" scorecard.py                   # WOE 评分卡 → scorecard.txt + 图 10/11
"$PY" clustering.py                  # 聚类分群 → clustering.txt + 图 12/13
"$PY" ab_test.py                     # A/B 实验与功效分析 → ab_test.txt + 图 14
"$PY" psm.py                         # PSM 因果推断 → psm.txt + 图 15
```

### 实测结论速览

**① 时间泄漏（run_fixed.log vs run_leaky.log）**

| 指标 | 修正版（无泄漏） | 含泄漏版 |
|---|---|---|
| 梯度提升 AUC | **0.7139** | 0.9996 |
| KS | **0.3193** | 0.9881 |
| PR-AUC | **0.4008** | 0.9990 |

> 差值 **0.286** 即时间泄漏造成的虚假提升。

**② 时间外验证（run_temporal.log）**

用 2014 年及以前训练、2015 年及以后测试（剔除 `issue_year` 特征）：

| 划分方式 | 训练集坏账率 | 测试集坏账率 | GBDT AUC |
|---|---|---|---|
| 随机划分 8:2 | 21.52% | 21.53% | **0.7139** |
| 时间外（< 2015） | 17.49% | 23.51% | **0.7053** |

> 随机划分高估约 **0.0086**。幅度不大，但方向是对的 —— 而且它暴露了更重要的
> 事实：**训练集 17.5% vs 测试集 23.5%**，坏账率本身随时间漂移，
> 随机划分会假装两个时期同分布。

**③ 成本敏感审批阈值（cost_analysis.txt）**

代价比 FN:FP = 6:1（LGD 60% vs 净利润率 10%）时：

| 审批阈值 | 期望总成本 | 通过率 | 拦截坏账 |
|---|---|---|---|
| 默认 0.500 | 22.40 百万 | 96.9% | 8.5% |
| **理论 fp/(fp+fn) = 0.143** | **11.04 百万** | 35.8% | **85.1%** |
| 实测扫描最优点 0.140 | 10.98 百万 | 34.0% | 86.5% |

> 理论解与实测最优点只差 **0.003** —— 这反过来证明模型概率是**校准可信**的
> （预测概率均值 21.48% vs 实际坏账率 21.53%）。
> 默认 0.5 几乎是最差的选择，比最优解贵一倍。

**④ WOE + IV 评分卡（scorecard.txt）**

| 模型 | AUC | PR-AUC | KS |
|---|---|---|---|
| 评分卡（4 变量 WOE + LR） | 0.6979 | 0.3690 | 0.3006 |
| 梯度提升（GBDT） | 0.7133 | 0.3981 | 0.3180 |

> 入选变量：`sub_grade`(IV 0.449)、`term`(0.184)、`verification_status`(0.049)、
> `home_ownership`(0.024)。评分十分位坏账率 **44.7% → 5.5% 完全单调**。
> AUC 低 **0.0154**，换来的是每一项得分都能逐条拆给业务方 —— 可解释性的代价。

**⑤ 聚类分群（clustering.txt）**

| k | 轮廓系数 | 坏账率极差 |
|---|---|---|
| **2**（轮廓系数最优） | **0.1271** | 0.64pp ← 几何最优但没有业务区分度 |
| **5**（业务指标选） | 0.0923 | **16.46pp** |

> 这是全项目最反直觉的一个发现：**轮廓系数选出的 k=2 在业务上是废的**
> （两群坏账率只差 0.64pp）。轮廓系数衡量的是**几何紧凑度，不是业务价值**。
> 改用坏账率极差选 k=5，各群坏账率 **11.3% ~ 27.8%**，且 K-Means 与
> 层次聚类(Ward) 独立跑出的跨度都在 15~16pp，结构稳定。
> 交叉验证：低风险群 54% 落在最优评级 A，但**其余四群评级分布几乎一样**、
> 坏账率却从 19.9% 拉到 27.8% —— 说明聚类捕捉到了**评级之外**的结构。

**⑥ A/B 实验设计与显著性检验（ab_test.txt）**

> ⚠️ 是**离线模拟实验**（随机对半分流），统计方法成立，但不等于真实线上 A/B。

| | 对照组 A（阈值 0.500） | 实验组 B（阈值 0.143） |
|---|---|---|
| 通过率 | 97.0% | 34.9% |
| 通过人群坏账率 | 20.35% | **8.86%** |

> 双样本比例 z 检验：坏账率差 **+11.50pp，z = 11.97，p ≈ 0**（95% CI [+9.91, +13.09]）。
> SRM 检查通过（分流比 0.9998，p = 0.99）。
> **两个指标都显著但方向相反** —— 统计只能说"差异是真的"，"值不值得我们上"
> 是业务决策。
>
> 功效分析：基线 20.35%、α=0.05、power=80% 下，检测 2pp 效应需每组 **6,125** 条；
> 现有每组 6,055 条 → 可检测 MDE = **2.01pp**。实测差异 11.50pp 远大于 MDE，
> 说明结论不是靠堆样本量堆出来的假显著。

**⑦ PSM 倾向得分匹配（psm.txt）**

因果问题：**「选 60 个月期限是不是导致违约率更高的原因？」**

| | 估计值 |
|---|---|
| 朴素比较（未调整） | +17.80pp |
| **PSM 匹配后（ATT）** | **+10.53pp**，bootstrap 95% CI [+9.03, +11.91] |

> 匹配掉混淆后效应**缩小 41%** —— 说明朴素比较里的差异，很大一部分来自
> "选长期限的人**本来**风险就高"，而不是期限本身。
> 平衡性：20/40 个协变量不平衡 → **0/40**，平均 |SMD| 由 0.160 降至 0.013。
> 卡钳 0.2×SD(logit PS)，1:1 无放回匹配，成功配对 7,423 对。
>
> ⚠️ PSM 只能控制**可观测**混淆；存在未观测混淆时仍有偏，属观察性研究证据，
> 强度低于随机实验。

> 完整数字卡片见 `简历与面经.md` 第五节。

---

## 六、进阶加分项（有余力再做）

- **时间外验证**：不用随机划分，改成"用 2015 年前的贷款训练，2015-2018 的测试"，
  更贴近真实上线场景。
- **评分卡视角**：用 WOE + IV 做特征筛选，逻辑回归输出评分卡——这是银行风控的标准做法。
- **成本敏感阈值**：不取 0.5 做阈值，而是按"漏掉一个坏账损失 X 元、误拒一个好客户损失 Y 元"
  求最优阈值。
