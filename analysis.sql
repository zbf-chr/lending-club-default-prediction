-- ===========================================================================
-- Lending Club 贷款违约 —— 核心指标 SQL 版
-- ===========================================================================
-- 目的：同一个业务结论，用 SQL 再表达一遍。
--
-- 为什么要有这个文件？
--   建模岗看的是 model pipeline；【数据分析岗看的是你能不能自己取数】。
--   面试官常问："你这个坏账率 21.5% 是怎么算出来的？" —— 你得能当场写出来。
--
-- 口径声明（与 pipeline.py / eda.py 完全一致）：
--   1. 坏账定义 bad_loan = 1：Charged Off / Default / Late(31-120 days)
--      及 "Does not meet the credit policy. Status:Charged Off"
--   2. 剔除结局未定：Current / Issued / In Grace Period / Late(16-30 days)
--   3. 表名 loans，由 run_sql.py 按 pipeline.clean() 准备好后注册
--
-- 执行方式：
--   "D:/venvs/lending-club/Scripts/python.exe" run_sql.py
-- ===========================================================================


-- ---------------------------------------------------------------------------
-- Q1｜基线：坏账率是多少？"全猜好账"的准确率又是多少？
-- ---------------------------------------------------------------------------
-- 这是全文所有指标的参照系。准确率 78% 听起来不错，
-- 但它其实是【什么都不做】就能拿到的分数 —— 这就是类别不平衡的坑。
SELECT
    COUNT(*)                                    AS "总记录数",
    SUM(bad_loan)                               AS "坏账笔数",
    ROUND(100.0 * AVG(bad_loan), 2)             AS "坏账率%",
    ROUND(100.0 * (1 - AVG(bad_loan)), 2)       AS "全猜好账的准确率%"
FROM loans;


-- ---------------------------------------------------------------------------
-- Q2｜评级有效性验证：grade 的坏账率是否单调？各档占多少样本？
-- ---------------------------------------------------------------------------
-- 窗口函数做三件事：排名、与全局基准的偏差、累计样本占比。
-- 如果 A→G 不是单调上升，说明要么标签定义错了，要么数据有问题。
WITH by_grade AS (
    SELECT
        grade,
        COUNT(*)        AS n,
        SUM(bad_loan)   AS bad_cnt,
        AVG(bad_loan)   AS bad_rate
    FROM loans
    GROUP BY grade
)
SELECT
    grade                                                   AS "评级",
    n                                                       AS "样本量",
    bad_cnt                                                 AS "坏账笔数",
    ROUND(100.0 * bad_rate, 2)                              AS "坏账率%",
    ROUND(100.0 * (bad_rate - (SELECT AVG(bad_loan) FROM loans)), 2)
                                                            AS "高于基准(pp)",
    RANK() OVER (ORDER BY bad_rate DESC)                    AS "风险排名",
    ROUND(100.0 * SUM(n) OVER (ORDER BY bad_rate DESC)
                / SUM(n) OVER (), 2)                        AS "样本累计占比%"
FROM by_grade
ORDER BY grade;


-- ---------------------------------------------------------------------------
-- Q3｜交叉分析：期限 × 评级，哪个格子最危险？
-- ---------------------------------------------------------------------------
-- 单变量看完了要看组合。面试常问"两个维度交叉你怎么呈现"——
-- 条件聚合（CASE WHEN + AVG）写成矩阵，比画十条线清楚。
-- 只保留样本量 >= 100 的格子，避免小样本噪声。
SELECT
    term                                                        AS "期限",
    COUNT(*)                                                    AS "样本量",
    ROUND(100.0 * AVG(bad_loan), 2)                             AS "整体",
    ROUND(100.0 * AVG(CASE WHEN grade IN ('A','B')       THEN bad_loan END), 2) AS "A-B",
    ROUND(100.0 * AVG(CASE WHEN grade IN ('C','D')       THEN bad_loan END), 2) AS "C-D",
    ROUND(100.0 * AVG(CASE WHEN grade IN ('E','F','G')   THEN bad_loan END), 2) AS "E-G"
FROM loans
GROUP BY term
ORDER BY AVG(bad_loan) DESC;


-- ---------------------------------------------------------------------------
-- Q4｜时间维度：按放款年份看坏账率，并算环比变化
-- ---------------------------------------------------------------------------
-- 为什么重要：【同一批贷款在不同年份的风险不一样】。
-- 这就是为什么验证不能简单随机划分 —— 随机划分会把未来信息漏给过去。
-- 真实上线必须是"用过去训练、预测未来"。
WITH by_year AS (
    SELECT
        issue_year,
        COUNT(*)        AS n,
        AVG(bad_loan)   AS bad_rate
    FROM loans
    WHERE issue_year IS NOT NULL
    GROUP BY issue_year
)
SELECT
    issue_year                                              AS "放款年份",
    n                                                       AS "样本量",
    ROUND(100.0 * bad_rate, 2)                              AS "坏账率%",
    ROUND(100.0 * (bad_rate - LAG(bad_rate) OVER (ORDER BY issue_year)), 2)
                                                            AS "环比变化(pp)",
    ROUND(100.0 * SUM(n) OVER (ORDER BY issue_year)
                / SUM(n) OVER (), 2)                        AS "样本累计占比%"
FROM by_year
ORDER BY issue_year;


-- ---------------------------------------------------------------------------
-- Q5｜★ 风险分层：把申请人按利率分成 10 层，看每层的坏账率与 lift
-- ---------------------------------------------------------------------------
-- 这是风控最标准的做法：不输出"某人坏账概率 0.03"，
-- 而是输出"某人落在第 8 层，该层历史坏账率 41%"——业务方听得懂。
--
-- lift = 该层坏账率 / 全局坏账率。lift > 1 表示这层风险高于平均。
-- 第 10 层和第 1 层的 lift 差距，就是这套分层的区分能力。
WITH scored AS (
    SELECT
        bad_loan,
        -- 用利率分 10 等份：利率是放款前可得的、且业务上直接体现定价风险
        NTILE(10) OVER (ORDER BY int_rate) AS bucket
    FROM loans
),
by_bucket AS (
    SELECT
        bucket,
        COUNT(*)        AS n,
        SUM(bad_loan)   AS bad_cnt,
        AVG(bad_loan)   AS bad_rate
    FROM scored
    GROUP BY bucket
)
SELECT
    bucket                                                  AS "风险层",
    n                                                       AS "样本量",
    bad_cnt                                                 AS "坏账笔数",
    ROUND(100.0 * bad_rate, 2)                              AS "坏账率%",
    ROUND(bad_rate / (SELECT AVG(bad_loan) FROM loans), 2)  AS "lift",
    ROUND(100.0 * n / SUM(n) OVER (), 2)                    AS "样本占比%",
    ROUND(100.0 * bad_cnt / SUM(bad_cnt) OVER (), 2)        AS "捕获坏账占比%"
FROM by_bucket
ORDER BY bucket;


-- ---------------------------------------------------------------------------
-- Q6｜★ 成本视角：如果只拒绝风险最高的一层，能拦下多少坏账、误伤多少好客户？
-- ---------------------------------------------------------------------------
-- 上一张图说"第 10 层最危险"，这张图回答业务问题："那我该拦到哪一层？"
-- 拦得越多，抓到的坏账越多，但误拒的好客户也越多 —— 这就是阈值权衡。
-- 面试时把这张表画出来，比说"我取了 0.5 阈值"高一个层次。
WITH scored AS (
    SELECT
        bad_loan,
        loan_amnt,
        NTILE(10) OVER (ORDER BY int_rate) AS bucket
    FROM loans
),
cum AS (
    SELECT
        bucket,
        COUNT(*)                AS n,
        SUM(bad_loan)           AS bad_cnt,
        SUM(1 - bad_loan)       AS good_cnt,
        SUM(CASE WHEN bad_loan = 1 THEN loan_amnt ELSE 0 END) AS bad_amt
    FROM scored
    GROUP BY bucket
)
SELECT
    bucket                                                          AS "拒绝至该层",
    ROUND(100.0 * SUM(bad_cnt)  OVER (ORDER BY bucket DESC)
                / SUM(bad_cnt)  OVER (), 2)                         AS "累计拦截坏账%",
    ROUND(100.0 * SUM(good_cnt) OVER (ORDER BY bucket DESC)
                / SUM(good_cnt) OVER (), 2)                         AS "累计误拒好客户%",
    ROUND(SUM(bad_amt) OVER (ORDER BY bucket DESC)
                / 1e6, 2)                                           AS "累计拦截本金(百万)",
    ROUND(100.0 * SUM(n) OVER (ORDER BY bucket DESC)
                / SUM(n) OVER (), 2)                                AS "累计拒绝率%"
FROM cum
ORDER BY bucket DESC;


-- ---------------------------------------------------------------------------
-- Q7｜缺失不是噪音：从未查过征信的人，坏账率反而更低
-- ---------------------------------------------------------------------------
-- 直接对应 pipeline 里"为什么不能无脑 fillna(0)"那个决策。
--
-- mths_since_recent_inq 缺失 = 这个人征信报告里【从来没有过查询记录】，
-- 而不是"0 个月前刚查询过"。无脑 fillna(0) 会把这批人写进"刚刚查征信"那一档,
-- 而刚刚频繁查征信恰恰是高风险信号 —— 语义正好填反了。
--
-- 第三列用「最近 1 个月内查询过的人」做对照：如果填 0，缺失组就被并进这一档。
SELECT
    mths_since_recent_inq_missing                   AS "从未查询过征信",
    COUNT(*)                                        AS "样本量",
    SUM(bad_loan)                                   AS "坏账笔数",
    ROUND(100.0 * AVG(bad_loan), 2)                 AS "坏账率%",
    ROUND(AVG(bad_loan) / (SELECT AVG(bad_loan) FROM loans), 2) AS "lift",
    ROUND(100.0 * AVG(CASE WHEN mths_since_recent_inq <= 1
                           THEN bad_loan END), 2)   AS "对照:最近1月内查询者%"
FROM loans
GROUP BY mths_since_recent_inq_missing
ORDER BY mths_since_recent_inq_missing;


-- ---------------------------------------------------------------------------
-- Q8｜帕累托：坏账本金集中在哪些评级/用途里？
-- ---------------------------------------------------------------------------
-- "80% 的坏账来自 20% 的客户"能不能成立，要用数据说话。
-- 按坏账本金降序排，算累计占比，看需要多少个格子才能覆盖 80%。
WITH by_seg AS (
    SELECT
        grade || ' / ' || purpose                       AS segment,
        SUM(CASE WHEN bad_loan = 1 THEN loan_amnt ELSE 0 END) AS bad_amt,
        COUNT(*)                                        AS n,
        AVG(bad_loan)                                   AS bad_rate
    FROM loans
    GROUP BY grade, purpose
    HAVING COUNT(*) >= 100
)
SELECT
    segment                                                 AS "评级/用途",
    n                                                       AS "样本量",
    ROUND(100.0 * bad_rate, 2)                              AS "坏账率%",
    ROUND(bad_amt / 1e6, 2)                                 AS "坏账本金(百万)",
    ROUND(100.0 * bad_amt / (SELECT SUM(bad_amt) FROM by_seg), 2)
                                                            AS "占坏账本金%",
    ROUND(100.0 * SUM(bad_amt) OVER (ORDER BY bad_amt DESC)
                / (SELECT SUM(bad_amt) FROM by_seg), 2)     AS "累计占比%"
FROM by_seg
ORDER BY bad_amt DESC
LIMIT 15;
