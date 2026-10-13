# -*- coding: utf-8 -*-
"""
Lending Club 贷款违约预测 —— 修正版流水线
=========================================

基于 Kaggle notebook `emmaruyiyang/lending-club-loan-default-prediction-eda` 重写。

核心业务问题
------------
在【放款前】的时点上，只用申请人当时就能拿到的信息，预测这笔贷款会不会违约。

原版有 4 个会让结果失真的问题（本文件全部修掉）：

  1) 标签泄漏    原版把 loan_condition / loan_status 留在特征矩阵里。
                实测：loan_condition_Good Loan 与标签相关系数 = -1.0
  2) 时间泄漏    原版用了 last_fico_range_* / last_pymnt_d 等"放款后"字段。
                其中 last_fico_range_* 恰好是原版相关性最高的两个特征(-0.66/-0.56)
  3) 特征选择失效 原版对 StandardScaler 之后的数据做 VarianceThreshold(threshold=1)。
                标准化后方差恒等于 1，判断条件 `var > 1` 等价于抛硬币
                （实测：40 个特征随机保留 19 个）
  4) Stacking 写错 原版用样本内预测训练元模型，分数虚高

另外修正：
  - CV 改用 StratifiedKFold + 固定随机种子
  - 重采样移到【每一折的训练部分内部】，避免 CV 在 1:1 平衡集上评估失真
  - 缺失值按业务语义填充（`mths_since_last_delinq` 缺失 = 从未逾期，填 0 语义相反）
  - 标签定义区分【确定坏 / 确定好 / 模糊剔除】三档

用法
----
    python pipeline.py --smoke     # 用合成数据跑通全流程（不需要数据集，先跑这个）
    python pipeline.py             # 修正版：用真实数据
    python pipeline.py --leaky     # 原版口径：故意保留泄漏字段，用来做对比实验
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

# Windows 控制台默认是 GBK，中文和特殊符号会乱码/报错，强制走 UTF-8
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

SEED = 42

# 数据集位置：按顺序找第一个存在的。
# .gz 压缩包可以直接读，pandas 会自动解压，不用手动解成 1.7G 的 CSV。
DATA_CANDIDATES = [
    Path("C:/Users/小芬/lending-club-project/data/accepted_2007_to_2018Q4.csv.gz"),
    Path("C:/Users/小芬/lending-club-project/data/accepted_2007_to_2018Q4.csv"),
    Path("D:/data/lending-club/accepted_2007_to_2018Q4.csv.gz"),
    Path("D:/data/lending-club/accepted_2007_to_2018Q4.csv"),
]
SAMPLE_N = 100_000          # 原版也抽样 10 万行；全量 220 万行你的内存扛不住
K_FEATURES = 20             # 折内选多少个特征
N_SPLITS = 5                # 交叉验证折数

# ---------------------------------------------------------------------------
# 一、业务定义：什么算"坏账"
# ---------------------------------------------------------------------------
# 这是整个项目最重要的业务判断，面试一定会问"你凭什么这么定义"。
# 三档分开处理，比原版"一股脑塞进坏账"更经得起推敲。
BAD_STATUS = {
    "Charged Off",                                      # 已核销 —— 确定的坏账
    "Default",                                          # 违约
    "Does not meet the credit policy. Status:Charged Off",
    "Late (31-120 days)",                               # 逾期 120 天以上，实质违约
}
GOOD_STATUS = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}
AMBIGUOUS_STATUS = {
    "Current",            # 还在还，不知道结局 —— 原版处理对了，剔除
    "Issued",
    "In Grace Period",    # 宽限期 / 逾期 16-30 天：结局未定，剔除最稳妥
    "Late (16-30 days)",
}

# ---------------------------------------------------------------------------
# 二、字段分组：哪些能用，哪些用了就是泄漏
# ---------------------------------------------------------------------------
ID_COLS = ["id", "member_id"]

# 直接用标签派生出来的，必须删
LEAK_COLS = ["loan_condition", "loan_condition_int", "loan_status"]

# 【放款后】才产生的字段 —— 用了就是时间泄漏，这是原版最大的问题
POST_ORIGINATION = [
    # 还款结果
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee",
    # 还款/征信事件时间
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d",
    # ⭐ 原版相关性最高的两个特征，都是贷款存续期间每月更新的 FICO
    "last_fico_range_high", "last_fico_range_low",
    # 放款后才出现的救助/处置计划
    "hardship_flag", "hardship_type", "hardship_reason", "hardship_status",
    "hardship_amount", "hardship_start_date", "hardship_end_date",
    "hardship_length", "hardship_dpd", "hardship_loan_status",
    "payment_plan_start_date", "deferral_term",
    "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount", "hardship_last_payment_amount",
    "debt_settlement_flag", "debt_settlement_flag_date", "settlement_status",
    "settlement_date", "settlement_amount", "settlement_percentage",
    "settlement_term",
    # 批准后确定
    "funded_amnt", "funded_amnt_inv",
]

# 高基数文本 / 常量列，没有泛化价值
JUNK_COLS = ["emp_title", "url", "title", "desc", "zip_code", "policy_code",
             "pymnt_plan", "application_type", "annual_inc_joint", "dti_joint",
             "verification_status_joint", "sec_app_fico_range_low",
             "sec_app_fico_range_high", "sec_app_earliest_cr_line",
             "sec_app_inq_last_6mths", "sec_app_mort_acc", "sec_app_open_acc",
             "sec_app_revol_util", "sec_app_open_act_il", "sec_app_num_rev_accts",
             "sec_app_chargeoff_within_12_mths",
             "sec_app_collections_12_mths_ex_med",
             "sec_app_mths_since_last_major_derog", "revol_bal_joint"]

# 这些字段"缺失"有业务含义：缺失 = 从未发生，而不是 0
# ⚠️ 原版统一 fillna(0)，等于说"这个人上个月刚逾期" —— 语义完全反了
NEVER_HAPPENED = ["mths_since_last_delinq", "mths_since_last_record",
                  "mths_since_last_major_derog", "mths_since_recent_bc",
                  "mths_since_recent_bc_dlq", "mths_since_recent_inq",
                  "mths_since_recent_revol_delinq"]

# 需要从 CSV 里读进来的列（用于 usecols，避免把 1.7G 全读进内存）
NUM_CANDIDATES = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti", "delinq_2yrs",
    "inq_last_6mths", "mths_since_last_delinq", "mths_since_last_record",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "acc_now_delinq", "tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim",
    "collections_12_mths_ex_med", "mths_since_last_major_derog",
    "fico_range_low", "fico_range_high", "open_acc_6m", "open_act_il",
    "open_il_12m", "open_il_24m", "mths_since_rcnt_il", "total_bal_il",
    "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc", "all_util",
    "total_cu_tl", "inq_fi", "inq_last_12m", "acc_open_past_24mths",
    "avg_cur_bal", "bc_open_to_buy", "bc_util", "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl",
    "mort_acc", "mths_since_recent_bc", "mths_since_recent_inq",
    "num_accts_ever_120_pd", "num_actv_bc_tl", "num_actv_rev_tl",
    "num_bc_sats", "num_bc_tl", "num_il_tl", "num_op_rev_tl", "num_rev_accts",
    "num_rev_tl_bal_gt_0", "num_sats", "num_tl_op_past_12m", "pct_tl_nvr_dlq",
    "percent_bc_gt_75", "pub_rec_bankruptcies", "tax_liens", "tot_hi_cred_lim",
    "total_bal_ex_mort", "total_bc_limit", "total_il_high_credit_limit",
]
CAT_CANDIDATES = ["term", "grade", "sub_grade", "home_ownership",
                  "verification_status", "purpose", "addr_state",
                  "initial_list_status", "emp_length"]
DATE_CANDIDATES = ["issue_d", "earliest_cr_line"]

RAW_COLS = NUM_CANDIDATES + CAT_CANDIDATES + DATE_CANDIDATES + ID_COLS \
           + POST_ORIGINATION + LEAK_COLS + JUNK_COLS


# ===========================================================================
# 工具函数
# ===========================================================================
def ks_stat(y_true, y_prob) -> float:
    """KS 统计量 = max(TPR - FPR)，信贷风控的核心指标。"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def recall_at_precision(y_true, y_prob, min_precision: float = 0.5) -> float:
    """在精确率不低于 min_precision 的前提下，能达到的最大召回率。
    比"阈值取 0.5 看召回"更贴近业务：先定我能接受的误伤率，再看能抓多少坏人。"""
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ok = prec >= min_precision
    return float(rec[ok].max()) if ok.any() else 0.0


def evaluate(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
        "KS": ks_stat(y_true, y_prob),
        "Acc": accuracy_score(y_true, y_pred),
        "P": precision_score(y_true, y_pred, zero_division=0),
        "R": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "R@P0.5": recall_at_precision(y_true, y_prob, 0.5),
    }


# ===========================================================================
# 缺失值处理：按业务语义填，而不是无脑 fillna(0)
# ===========================================================================
def impute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 1) 缺失 = "从未发生" 的字段：填一个哨兵值，并加缺失指示列
    for col in NEVER_HAPPENED:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isnull().astype(int)
            df[col] = df[col].fillna(999)   # 999 个月 ≈ 从未发生
    # 2) 其余数值列取中位数（比均值抗离群）
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    # 3) 类别列填 "Unknown"
    #    注意：这里不能用 dtype == "object" 判断！pandas 3.0 起默认字符串类型是
    #    `str` 而不是 `object`，用 is_numeric_dtype 才是跨版本安全的写法。
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")
    return df


# ===========================================================================
# Target Encoding —— 必须用 out-of-fold，否则等于拿标签编码自己
# ===========================================================================
def _smooth_map(codes, y, prior: float, smoothing: float) -> pd.Series:
    st = pd.DataFrame({"c": codes, "y": y}).groupby("c")["y"].agg(["mean", "count"])
    return (st["mean"] * st["count"] + prior * smoothing) / (st["count"] + smoothing)


def target_encode(X_tr, y_tr, X_te, cat_cols, smoothing: float = 20.0,
                  oof: bool = True):
    """对多分类变量做（可选 out-of-fold 的）target encoding。

    oof=True ：训练集用「别的折的统计量」编码，避免过拟合（交叉验证时用）
    oof=False：训练集直接用全部训练数据编码（训练最终模型时用，测试集始终只用训练集统计量）
    """
    prior = float(np.mean(y_tr))
    X_tr, X_te = X_tr.copy(), X_te.copy()
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for col in cat_cols:
        full_map = _smooth_map(X_tr[col].values, y_tr.values, prior, smoothing)
        X_te[col] = pd.Series(X_te[col].values).map(full_map).fillna(prior).values

        if oof:
            enc = np.full(len(X_tr), prior, dtype=float)
            for tr_pos, _ in inner.split(X_tr, y_tr):
                m = _smooth_map(X_tr[col].values[tr_pos], y_tr.values[tr_pos],
                                prior, smoothing)          # 只用这一折的统计量
                enc[tr_pos] = pd.Series(X_tr[col].values[tr_pos]).map(m) \
                                .fillna(prior).values
            X_tr[col] = enc
        else:
            X_tr[col] = pd.Series(X_tr[col].values).map(full_map) \
                          .fillna(prior).values
    return X_tr, X_te


# ===========================================================================
# 重采样：只在【训练折内部】做，不能在 CV 之前做
# ===========================================================================
def undersample(X, y, seed: int = SEED):
    """随机欠采样到 1:1。

    ⚠️ 注意它的副作用：模型输出的概率会被校准到 50/50 先验上。
    如果下游还要接 stacking / 做阈值决策，这个偏移会坑你（原版 notebook 就中招了：
    它的 Stacking 元模型 F1 只有 0.06）。所以默认不用它，改用 class_weight。
    """
    rng = np.random.default_rng(seed)
    pos_bad = np.flatnonzero(y.values == 1)
    pos_good = np.flatnonzero(y.values == 0)
    n = min(len(pos_bad), len(pos_good))
    keep = np.sort(np.concatenate([rng.choice(pos_bad, n, replace=False),
                                   rng.choice(pos_good, n, replace=False)]))
    return X.iloc[keep], y.iloc[keep]


def class_weight_vector(y) -> np.ndarray:
    """给少数类更高权重，但【不改变样本本身的分布】。
    代价是模型内部更关注坏账，好处是输出的概率仍然校准在真实先验上，
    下游 stacking 和阈值决策都不会错位。这是处理不平衡更稳妥的做法。"""
    counts = np.bincount(np.asarray(y).astype(int))
    w = len(y) / (len(counts) * counts)
    return w[np.asarray(y).astype(int)]


def fit_model(model, X, y, balance: str):
    m = clone(model)
    if balance == "classweight":
        return m.fit(X, y, sample_weight=class_weight_vector(y))
    return m.fit(X, y)


def select_features(X: pd.DataFrame, y, k: int = K_FEATURES) -> list:
    """折内用互信息选特征。注意：这一步必须只在训练折上做。"""
    mi = mutual_info_classif(X.values, y.values, random_state=SEED)
    order = np.argsort(mi)[::-1]
    return list(X.columns[order[:k]])


# ===========================================================================
# 数据加载与清洗
# ===========================================================================
def resolve_data_path(explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"--data 指定的文件不存在：{p}")
        return p
    for p in DATA_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "找不到数据集。请按 README 第二节下载，放到下面任一位置：\n  "
        + "\n  ".join(str(p) for p in DATA_CANDIDATES)
        + "\n或用 --data 显式指定路径，也可以先用 --smoke 跑通流程。")


def load_real_data(path: Path) -> pd.DataFrame:
    keep = set(RAW_COLS)
    # 只读需要的列，否则 2.2M 行 × 151 列会把内存打满
    df = pd.read_csv(path, usecols=lambda c: c in keep, low_memory=False)
    print(f"[加载] {path.name}：全量 {df.shape[0]:,} 行 × {df.shape[1]} 列")
    if len(df) > SAMPLE_N:
        df = df.sample(n=SAMPLE_N, random_state=SEED).reset_index(drop=True)
        print(f"[加载] 抽样至 {len(df):,} 行（random_state={SEED}）")
    return df


def make_synthetic_data(n: int = 8000) -> pd.DataFrame:
    """合成数据：只为在没有数据集时先把流程跑通，验证代码没写错。"""
    rng = np.random.default_rng(SEED)
    df = pd.DataFrame(index=range(n))
    for col in NUM_CANDIDATES:
        df[col] = np.round(rng.gamma(2.0, 8.0, n), 2)
    df["annual_inc"] = np.round(rng.lognormal(11, 0.5, n))
    df["fico_range_low"] = rng.integers(600, 800, n).astype(float)
    df["fico_range_high"] = df["fico_range_low"] + 4
    df["int_rate"] = np.round(rng.uniform(5, 28, n), 2)
    df["dti"] = np.round(rng.uniform(0, 40, n), 2)
    df["revol_util"] = np.round(rng.uniform(0, 100, n), 2)
    for col in CAT_CANDIDATES:
        df[col] = rng.choice(["A", "B", "C", "D", "E"], n)
    df["term"] = rng.choice([" 36 months", " 60 months"], n)
    df["emp_length"] = rng.choice(list(EMP_MAP.keys()), n)
    df["addr_state"] = rng.choice(["CA", "TX", "NY", "FL", "IL"], n)
    df["issue_d"] = rng.choice(["Jan-2015", "Jun-2016", "Mar-2017", "Nov-2018"], n)
    df["earliest_cr_line"] = rng.choice(["Jan-1995", "Aug-2001", "Mar-2008"], n)
    df["id"] = np.arange(n)
    df["member_id"] = np.arange(n)

    # 造一点真实信号：FICO 低 / 利率高 / 逾期多 -> 更容易坏账
    logit = (-2.0
             + 0.02 * (700 - df["fico_range_low"])
             + 0.10 * (df["int_rate"] - 15)
             + 0.35 * (df["dti"] - 15) / 10
             + 0.6 * rng.normal(0, 1, n))
    p_bad = 1 / (1 + np.exp(-logit))
    is_bad = rng.random(n) < p_bad
    ambiguous = rng.random(n) < 0.25
    df["loan_status"] = np.where(
        ambiguous, rng.choice(list(AMBIGUOUS_STATUS), n),
        np.where(is_bad, "Charged Off", "Fully Paid"))
    return df


EMP_MAP = {"10+ years": 10, "9 years": 9, "8 years": 8, "7 years": 7,
           "6 years": 6, "5 years": 5, "4 years": 4, "3 years": 3,
           "2 years": 2, "1 year": 1, "< 1 year": 0.5, "n/a": 0}

STATE_REGION = {
    "CA": "West", "OR": "West", "UT": "West", "WA": "West", "CO": "West",
    "NV": "West", "AK": "West", "MT": "West", "HI": "West", "WY": "West",
    "ID": "West", "AZ": "SouthWest", "TX": "SouthWest", "NM": "SouthWest",
    "OK": "SouthWest", "GA": "SouthEast", "NC": "SouthEast", "VA": "SouthEast",
    "FL": "SouthEast", "KY": "SouthEast", "SC": "SouthEast", "LA": "SouthEast",
    "AL": "SouthEast", "WV": "SouthEast", "DC": "SouthEast", "AR": "SouthEast",
    "DE": "SouthEast", "MS": "SouthEast", "TN": "SouthEast", "IL": "MidWest",
    "MO": "MidWest", "MN": "MidWest", "OH": "MidWest", "WI": "MidWest",
    "KS": "MidWest", "MI": "MidWest", "SD": "MidWest", "IA": "MidWest",
    "NE": "MidWest", "IN": "MidWest", "ND": "MidWest", "CT": "NorthEast",
    "NY": "NorthEast", "PA": "NorthEast", "NJ": "NorthEast", "RI": "NorthEast",
    "MA": "NorthEast", "MD": "NorthEast", "VT": "NorthEast", "NH": "NorthEast",
    "ME": "NorthEast",
}


def clean(df: pd.DataFrame, leaky: bool = False) -> pd.DataFrame:
    """清洗 + 派生特征。leaky=True 时故意保留泄漏字段（用于对比实验）。"""
    df = df.copy()

    # --- 1. 打标签，剔除结局未定的记录 -------------------------------------
    before = len(df)
    df = df[df["loan_status"].isin(BAD_STATUS | GOOD_STATUS)].copy()
    df["bad_loan"] = df["loan_status"].isin(BAD_STATUS).astype(int)
    print(f"[目标] 剔除 {before - len(df):,} 条结局未定记录，"
          f"剩余 {len(df):,} 条，坏账率 {df['bad_loan'].mean():.2%}")

    # --- 2. 删列 -----------------------------------------------------------
    drop = set(ID_COLS) | set(LEAK_COLS) | set(JUNK_COLS)
    if not leaky:
        drop |= set(POST_ORIGINATION)
    df = df.drop(columns=[c for c in drop if c in df.columns])

    # --- 3. 缺失太多的列直接丢（沿用原版 80% 阈值）--------------------------
    df = df.dropna(axis=1, thresh=int(0.8 * len(df)))

    # --- 4. 派生特征 -------------------------------------------------------
    if "emp_length" in df.columns:
        df["emp_length_int"] = df["emp_length"].map(EMP_MAP)
        df = df.drop(columns=["emp_length"])
    if "addr_state" in df.columns:
        df["region"] = df["addr_state"].map(STATE_REGION).fillna("Other")
        df = df.drop(columns=["addr_state"])
    if {"issue_d", "earliest_cr_line"}.issubset(df.columns):
        df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
        df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"],
                                                format="%b-%Y", errors="coerce")
        df["credit_hist_months"] = ((df["issue_d"] - df["earliest_cr_line"])
                                    .dt.days / 30.44).round(1)
        df["issue_year"] = df["issue_d"].dt.year
        df = df.drop(columns=["issue_d", "earliest_cr_line"])

    # --- 5. 按业务语义补缺失 ------------------------------------------------
    df = impute(df)
    return df


def assert_no_leakage(X: pd.DataFrame, y: pd.Series) -> None:
    """建模前的最后一道防线：任何与标签近乎完美相关的特征都要报警。"""
    bad = []
    for col in X.select_dtypes(include=[np.number]).columns:
        if X[col].nunique() < 2:
            continue
        c = abs(np.corrcoef(X[col].values, y.values)[0, 1])
        if c > 0.95:
            bad.append((col, round(float(c), 4)))
    if bad:
        raise ValueError(f"检测到疑似标签泄漏字段，请检查：{bad}")
    print("[检查] 未发现与标签高度相关的可疑字段 [OK]")


# ===========================================================================
# 建模
# ===========================================================================
def build_models() -> dict:
    return {
        "逻辑回归": LogisticRegression(max_iter=2000),
        "决策树": DecisionTreeClassifier(max_depth=6, random_state=SEED),
        "随机森林": RandomForestClassifier(n_estimators=200, max_depth=10,
                                           n_jobs=-1, random_state=SEED),
        "梯度提升": GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                               random_state=SEED),
    }
    # 想加 XGBoost / LightGBM 就取消下面注释（你机器上已装好）
    #   from xgboost import XGBClassifier
    #   from lightgbm import LGBMClassifier
    #   "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, verbosity=0,
    #                            random_state=SEED),
    #   "LightGBM": LGBMClassifier(n_estimators=200, max_depth=4, verbose=-1,
    #                              random_state=SEED),


def cross_validate(X, y, models, cat_cols, num_cols, balance: str = "classweight"):
    """分层 CV。每一折内部独立完成：target encoding -> 平衡 -> 标准化 -> 选特征。

    关键点：所有"会看到标签"的步骤（编码、选特征、标准化）都只在训练折上做，
    验证折保持真实的坏账先验，这样算出来的分数才是可信的。
    """
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = {name: np.zeros(len(X)) for name in models}
    records = []

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X.iloc[tr].copy(), X.iloc[va].copy()
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        # 1) target encoding（只用训练折）
        X_tr, X_va = target_encode(X_tr, y_tr, X_va, cat_cols, oof=True)
        # 2) 平衡（欠采样只在训练折内做；验证折永远保持真实先验）
        if balance == "undersample":
            X_tr, y_tr = undersample(X_tr, y_tr)
        # 3) 标准化（只用训练折 fit）
        num = [c for c in num_cols if c in X_tr.columns]
        scaler = StandardScaler().fit(X_tr[num])
        X_tr[num] = scaler.transform(X_tr[num])
        X_va[num] = scaler.transform(X_va[num])
        # 4) 特征选择（只用训练折）
        keep = select_features(X_tr, y_tr)
        X_tr, X_va = X_tr[keep], X_va[keep]

        for name, model in models.items():
            m = fit_model(model, X_tr, y_tr, balance)
            p = m.predict_proba(X_va)[:, 1]
            oof[name][va] = p
            records.append({"fold": fold, "model": name,
                            **evaluate(y_va, p)})
        print(f"  fold {fold}/{N_SPLITS} 完成")

    return pd.DataFrame(records), oof


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="用合成数据跑通流程，不需要数据集")
    parser.add_argument("--leaky", action="store_true",
                        help="原版口径：故意保留泄漏字段，用于对比实验")
    parser.add_argument("--balance", choices=["classweight", "undersample", "none"],
                        default="classweight",
                        help="不平衡处理方式（默认 classweight，原版用的是 undersample）")
    parser.add_argument("--data", default=None,
                        help="数据集路径（.csv 或 .csv.gz 都行），不填则自动查找")
    args = parser.parse_args()

    mode = "原版口径(含泄漏)" if args.leaky else "修正版(无泄漏)"
    print(f"\n{'=' * 60}\n运行模式：{mode}\n{'=' * 60}\n")

    # ---------- 数据 ----------
    df = make_synthetic_data() if args.smoke else load_real_data(resolve_data_path(args.data))
    df = clean(df, leaky=args.leaky)

    X = df.drop(columns=["bad_loan"])
    y = df["bad_loan"].reset_index(drop=True)
    X = X.reset_index(drop=True)
    # pandas 3.0 起字符串列的 dtype 是 `str` 而非 `object`，用 is_numeric_dtype 判断
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    print(f"[特征] 数值 {len(num_cols)} 列，类别 {len(cat_cols)} 列")

    assert_no_leakage(X, y)

    # ---------- 划分：留存集保持真实先验 ----------
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, te_idx = next(splitter.split(X, y))
    X_tr, X_te = X.iloc[tr_idx].reset_index(drop=True), X.iloc[te_idx].reset_index(drop=True)
    y_tr, y_te = y.iloc[tr_idx].reset_index(drop=True), y.iloc[te_idx].reset_index(drop=True)
    print(f"[划分] 训练 {len(X_tr):,} / 留存 {len(X_te):,}，"
          f"留存集坏账率 {y_te.mean():.2%}")

    # ---------- 交叉验证 ----------
    models = build_models()
    print(f"\n[交叉验证] {N_SPLITS} 折（验证折保持真实先验）｜不平衡处理：{args.balance}")
    cv_df, oof = cross_validate(X_tr, y_tr, models, cat_cols, num_cols,
                                balance=args.balance)

    summary = cv_df.groupby("model")[["AUC", "PR-AUC", "KS", "R", "R@P0.5"]] \
                   .mean().sort_values("AUC", ascending=False)
    print("\n[CV 结果]")
    print(summary.round(4).to_string())

    # 基线：全猜"好账"
    baseline_acc = 1 - y_te.mean()
    print(f"\n[基线] 全部预测为好账的准确率 = {baseline_acc:.2%}"
          f"（任何模型不显著超过这个数就没价值）")

    # ---------- Stacking：用 out-of-fold 预测训练元模型 ----------
    meta_X = pd.DataFrame({name: oof[name] for name in models})
    meta = LogisticRegression(max_iter=1000).fit(meta_X, y_tr)

    # ---------- 留存集评估 ----------
    X_tr2, X_te2 = target_encode(X_tr.copy(), y_tr, X_te.copy(), cat_cols, oof=False)
    scaler = StandardScaler().fit(X_tr2[num_cols])
    X_tr2[num_cols] = scaler.transform(X_tr2[num_cols])
    X_te2[num_cols] = scaler.transform(X_te2[num_cols])
    keep = select_features(X_tr2, y_tr)
    X_tr2, X_te2 = X_tr2[keep], X_te2[keep]

    # 欠采样模式下，最终模型也拟合在平衡后的训练集上（这是原版的做法）
    X_fit, y_fit = (undersample(X_tr2, y_tr)
                    if args.balance == "undersample" else (X_tr2, y_tr))

    holdout = {}
    meta_preds = []
    for name, model in models.items():
        m = fit_model(model, X_fit, y_fit, args.balance)
        p = m.predict_proba(X_te2)[:, 1]
        meta_preds.append(p)
        holdout[name] = evaluate(y_te, p)
    holdout["Stacking(元模型)"] = evaluate(
        y_te, meta.predict_proba(np.column_stack(meta_preds))[:, 1])

    result = pd.DataFrame(holdout).T.sort_values("AUC", ascending=False)
    print(f"\n[留存集结果 · {mode}]")
    print(result.round(4).to_string())
    nxt = "" if args.leaky else "--leaky"
    print(f"\n[对比提示] 记下这里的 AUC，再跑一次 `python pipeline.py {nxt}` "
          "对比两者差距 —— 这个差值就是【泄漏带来的虚假提升】。")
    print("\n选中的特征：", keep)


if __name__ == "__main__":
    main()
