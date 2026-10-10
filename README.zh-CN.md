# Annotation Efficient NER for Industrial Maintenance

## 当前目标

先建立 MaintIE 公开基准：在相同原始 token 标注预算下，比较 Random、Uncertainty、Uncertainty + Diversity。模型通过 BERT/RoBERTa/ModernBERT feasibility pilot 后选定，采样比较中保持一致。主指标为 exact-match entity-level micro-F1，至少五个随机种子。

本目录是 2026-10-10 从零启动的项目。依据用户提供的 proposal 建立；已完成三种编码器的可行性试跑，主动学习采样尚未实现。

## 已完成

- 保存官方 gold 数据原件和 MIT 许可证。
- 提供可重复运行的数据审计：记录数、原始 token 数、五类实体频次、重复文本、实体边界和重叠检查。
- 发现 328 条记录存在重叠实体，暂停 BIO 转换，保留全部原始标注。

## 第一次检查结果

共 1,076 条记录、6,119 个原始 token、3,397 个实体标注。五类实体频次：PhysicalObject 1,994；Activity 784；State 438；Process 146；Property 35。没有非法实体边界；有一组重复文本（零基索引 874、875）。这些是完整原始数据的统计，尚不是训练池预算。

328 条记录包含重叠实体。例如 `air filters` 与 `filters` 同时被标注。官方指南 §2.3.6 明确允许 nested entities。普通单层 BIO 无法表示全部原始实体；脚本检测到重叠时只写审计报告，暂不输出 BIO。

已选择 span-based NER 路线：保留全部嵌套实体，通过相同的 span 分类头比较 BERT/RoBERTa/ModernBERT 编码器。共用 span 分类头已实现，三种模型均完成训练与 checkpoint 重载检查；设计见 `docs/model_plan.md`。

## 数据与评估准备已完成

- 固定 split seed：20261010。按重复/近重复文本连通分组后，以分组数约 70/15/15 划分；原始记录比例会略有偏差。
- 当前训练 755 条（4,313 tokens），验证 155 条（853 tokens），测试 166 条（953 tokens）。原始 3,397 个实体全部保留。
- 完全一致的大小写归一化 token 序列，以及 token SequenceMatcher 相似度 >=0.9 的文本留在同一集合。此规则只能控制表面相似泄漏，不能证明记录来自不同设备或不同工单，因为源数据缺少这些标识。
- 验证集仅有 4 个 Property 实体，测试集有 9 个；该类别结果应谨慎解释，不以该类别的单轮 F1 作为独立停止条件。
- 实体级评分支持嵌套；要求记录索引、起点、终点、类别全部一致才算正确。
- 12 项本地测试通过（其中 3 项需要训练依赖）；全数据转换逐条核对原始标注，oracle 自评分为 1.0。这只验证数据和评分管线，不是模型性能。

```sh
python3 scripts/prepare_data.py
python3 -m unittest discover -s tests -v
python3 scripts/score_spans.py --gold data/processed/validation.jsonl --predictions YOUR_PREDICTIONS.jsonl
```

预测文件每条记录必须包含 `record_index` 和 `entities`；未预测到实体的记录也应写入空列表。实体格式：`{"start": 0, "end": 2, "type": "PhysicalObject"}`，end 为 exclusive token offset。

官方标注指南：https://github.com/nlp-tlp/maintie/blob/main/GUIDELINES.md

## 运行数据检查

在本目录运行（仅使用 Python 标准库）：

```sh
python3 scripts/audit_data.py
```

`data/raw/` 保留原始数据；`reports/data_audit.json` 包含原始文件 SHA-256 和检查结果。仅在没有重叠或非法实体时，脚本才会生成 `data/processed/` 下的 BIO 数据。

## 模型试跑结果

事前固定相同数据、共用分类头、seed 42、5 轮预算：BERT 验证集 micro-F1 0.9072，RoBERTa 0.8936，ModernBERT 0.8669。按注册规则，后续工作模型选用 BERT-base-uncased。测试集未用于选型。这是单种子短预算试跑，不能据此断言 BERT 普遍更优。

完整结果：[pilot_results.md](docs/pilot_results.md)。规则：[pilot_protocol.md](docs/pilot_protocol.md)。

## 下一步：主动学习比较

1. 已完成固定数据划分及实体级评分。
2. 建立独立训练环境，检查原始 token 与 subword 对齐。本机检测到 PyTorch 2.14.0、Apple MPS 可用、CUDA 不可用；三种模型均已在 MPS 上完成试跑。
3. 已完成候选模型 pilot，选定 BERT-base-uncased 为工作编码器。测试集不参与模型选择或调参。
4. 建立全量训练参照及 Random 曲线，再接入两种主动学习策略。

## 实验协议中仍需明确

- 初始随机标注集、每轮预算、整条记录无法精确匹配 token 预算时的规则。
- Uncertainty 的序列聚合方式；Hybrid 的候选池大小、表示与距离。
- 每轮从原始预训练权重重训，还是继续训练；所有策略采用一致规则。
- 5/10/20/40/60/100% 预算以训练池原始 token 总量计算；记录实际获取的 token 数。
- 用验证集制定停止规则；测试集用于冻结协议后的报告，不依据测试曲线调整策略。
- 模拟标注减少的 token 数不能直接当作实际专家工时节省。

## 范围与来源

主实验采用 gold corpus，映射为 PhysicalObject、State、Process、Activity、Property 五类。Silver corpus 暂不加入主实验。AZ 扩展依赖数据和标注批准；目前只进行公开数据工作。

官方仓库：https://github.com/nlp-tlp/maintie

论文：https://aclanthology.org/2024.lrec-main.954/

引用：Bikaun et al. (2024), MaintIE: A Fine-Grained Annotation Schema and Benchmark for Information Extraction from Maintenance Short Texts, LREC-COLING, 10939–10951.
