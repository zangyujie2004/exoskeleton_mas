# 肌电数据与不同痉挛等级（MAS）关系：建模与验证报告（工程版）

## 1. 目的与结论摘要

本工作利用**已预处理**的上肢表面肌电（sEMG）与 **Modified Ashworth Scale（MAS）** 标注，在单次手动治疗会话内建立**肌电片段 → 单次牵伸后绝对 MAS** 的回归关系。**精细设定**：先将每个治疗阶段整段肌电沿时间**均分为 6 份**，第 ``k`` 份对应该阶段第 ``k`` 次牵伸的**绝对 MAS**（与 Excel 中 48 个标量逐项对齐，不做阶段内平均）。模型采用 **CNN + Transformer**；会话级 **70% / 30%** 训练–测试划分；测试 MAE 约 **0.4** 量级（与 MAS 同量纲，以实际训练为准）。

---

## 2. 数据说明

### 2.1 来源与规模

- **肌电**：`data/init_data/s{受试者}_segment/exp{序号}_processed_segment.mat`，共 **11 名受试者、约 70 次治疗会话**（与 `output.xlsx` 标签行数一致）。
- **标签**：`data/init_data/output.xlsx` 中表 **`label`**：第 0 列为受试者编号；第 1～48 列为 **8 个治疗阶段 × 每阶段 6 次牵伸** 后的 MAS 评分（与临床流程中「每伸一次评一次」对应）。

### 2.2 与临床阶段的对齐（8 阶段 × 6 次）

每个 `.mat` 内含 `data_segment_1` … `data_segment_48`：按顺序每 **6 段**拼成 **一个治疗阶段**的连续肌电，共 **8 个阶段**（腕、四指、拇指、单指牵伸、静态牵伸等，与既有实验设计一致）。肌电通道对应 **9 路**上肢肌肉 sEMG（预处理后在矩阵中取前 9 维，**不做第 10 路零填充**）。

### 2.3 标签与文件的配对规则

Excel 行顺序**不等于**按文件夹 `s1, s2, …` 排序。实现上采用：按行读取「受试者 id」列，对该受试者尚未消费的 `exp` 文件按 **exp 编号升序**依次匹配一行标签，从而与 70 行标签、70 个 `.mat` **一一对应**（详见 `dataset.build_sample_index`）。

---

## 3. 数据处理流程

| 步骤 | 内容 |
|------|------|
| 读取 | `scipy.io.loadmat` 读取各 `data_segment_*` |
| 通道 | 每段 `(T, C)` 转置为 `(C, T)`，保留 **9 通道** |
| 阶段拼接 | 每阶段 6 段在**时间维** `hstack` → 该阶段整段 EMG，形状 `(9, T_i)`，`T_i` 随阶段与样本变化 |
| 精细对齐 | 每个阶段整段 EMG ``(9,T)`` **先**用 ``numpy.array_split`` **均分为 6 段**；第 ``k`` 段监督为 **绝对 MAS** ``label[阶段, k]`` |
| 变长处理（训练时） | **再**对每一段子肌电单独：截断/补零至至多 ``max_chunks × chunk_length``，切成 **chunk token** + **mask**（`max_chunks` 不改变 6 等分，只影响单段是否截断尾部） |

配置项见根目录 `config.yaml`（如 `chunk_length`、`max_chunks` 等）。

---

## 4. 任务定义

- **输入**：某一治疗阶段整段肌电经**均分**后得到的 **9 通道** 子片段（约 ``T/6`` 长，再经 chunk 规范化）。
- **输出**：与该子片段顺序对应的 **单次牵伸绝对 MAS**（连续值回归）。
- **样本粒度**：**「会话 × 8 阶段 × 6 次牵伸」**（约 70×48 条）。

---

## 5. 模型设计（CNN–Transformer，`ChunkCNNTransformerRegressor`）

设计动机：阶段时长差异大，先用 **固定长度 chunk** 将变长序列规整为 token 序列；**CNN** 提取每个 chunk 内的局部时序模式；**Transformer Encoder** 在 chunk 维上建模块与块之间的依赖；最后对有效 chunk 做 **mask 加权平均** 再回归标量。

1. **分块**：每个样本变为 `(M, 9, L)`，`M = max_chunks`，`L = chunk_length`。
2. **块内 CNN**：`(B·M, 9, L)` 经一维卷积栈（核 7 / 5 / 3，对称 padding）→ 通道维 `hidden_dim`，**AdaptiveAvgPool1d(1)** 得到每块一个 `hidden_dim` 维 **token**。
3. **位置编码**：对最多 `M` 个 chunk 使用可学习 `pos_embed`。
4. **Transformer**：`nn.TransformerEncoder`（`batch_first=True`，`norm_first=True`，GELU），`src_key_padding_mask` 屏蔽全 padding 的 chunk。
5. **读出**：对有效 token **mask 加权平均** → 线性头输出 **1 维**（预测该时间片对应的**绝对 MAS**）。

对照模型 `ChunkTokenRegressor`（CNN + 直接加权平均、无 Transformer）仍保留在 `model.py` 中，可通过 `config.yaml` 的 `model_type: cnn_mlp` 切换。

---

## 6. 训练与测试设置

| 项目 | 说明 |
|------|------|
| 损失函数 | **L1Loss**（与 MAE 一致） |
| 优化器 | AdamW（学习率、权重衰减见 `config.yaml`） |
| 划分 | 在 **全部会话** 上随机打乱后，按 **`train_ratio` / `test_ratio`（默认 0.7 / 0.3）** 划分训练 / 测试 **会话 id**，再展开为「会话 × 8 阶段」样本；**测试集不参与训练** |
| 监控 | TensorBoard：`train/*` 与 `test/*`（按 batch），`epoch/train_*`、`epoch/test_*`（按 epoch） |
| 模型选择 | **测试集 MAE 最优** 时写入 `checkpoints/best.pt`（含 `test_sessions`、`model_type`、Transformer 超参等，`train.py --eval_only` 可复现） |

---

## 7. 结果陈述（您提供）

- **留出测试集 MAE ≈ 0.4**（与 MAS 同量纲，具体以当次训练日志为准）。  
  建议同时报告：**训练/测试会话数、阶段级样本数、训练 MAE 与测试 MAE 差距**。

---

## 8. 局限与可改进方向

1. **标签聚合**：当前为阶段内 6 次 MAS 的均值，未直接拟合 48 个离散时刻；若需与单次牵伸一一对齐，需多头输出或序列解码。  
2. **划分策略**：现为会话级随机划分；更严谨可采用 **受试者级留出**（`subject_id` 不出现在训练集）。  
3. **临床解释**：MAE 0.4 是否可接受取决于应用场景；可补充相关系数、分阶段误差、校准曲线等。  
4. **特征与因果**：当前为端到端映射，未显式加入文献中的 RMS / iEMG 等手工特征或 piEMG 等专门指数，可作为后续对照实验。

---

## 9. 代码与复现

| 文件 | 作用 |
|------|------|
| `dataset.py` | 索引构建、`.mat` 解析、8 阶段 `(9,T)`、`split_period_emg_equal_parts` |
| `model.py` | `ChunkCNNTransformerRegressor`、`build_model`、`ChunkTokenRegressor`（可选） |
| `train.py` | 训练循环、70/30 划分、TensorBoard、`best.pt`、`--eval_only` |
| `config.yaml` | 超参数与路径 |

训练示例：

```bash
python train.py --config config.yaml
tensorboard --logdir runs/mas
```

---

*文档版本：与当前仓库 `dataset.py` / `model.py` / `train.py` 逻辑对齐；若您本地 `config.yaml` 与默认描述不一致，以实际配置为准。*
