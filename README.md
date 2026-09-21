# 地址抽取与对齐 (Address Alignment)

## 项目简介

基于**预训练语言模型**的中文地址实体识别与对齐工具。

从非结构化的中文地址文本中，自动抽取 **省 / 市 / 区 / 镇 / 详细地址 / 姓名 / 电话** 七个字段，
再借助 MySQL 中的**标准地址库**做模糊匹配 + 编辑距离校验，把模型预测结果纠偏为规范化地址。

项目来源于尚硅谷大模型技术实战课程，在其基础上完成了数据处理、模型微调、地址库对齐与
Web 服务化的完整链路整合。

---

## 技术栈

| 类别 | 选型 |
|---|---|
| 语言 / 框架 | Python 3.8+、PyTorch、HuggingFace Transformers / Datasets |
| 预训练模型 | `roberta-small-wwm-chinese-cluecorpussmall`（中文 RoBERTa，全词掩码） |
| 序列标注 | `RobertaForTokenClassification` / `BertForTokenClassification` + `Trainer` |
| 地址库 | MySQL（`region` 表存储省市区标准 `full_name`） |
| 模糊匹配 | RapidFuzz（编辑距离相似度） |
| Web 服务 | FastAPI + Uvicorn |

---

## 标签体系

共 **25 个标签**，采用 BIEOS 标注：

```
O
B/I/E-prov      省          B/I/E-city     市
B/I/E/S-district 区/县      B/I/E/S-town   镇/街道
B/I/E/S-detail   详细地址    B/I/E-name     姓名
B/I/E-phone      电话
```

标注定义见 `config.LABELS`。

---

## 核心流程

### 1. 数据预处理（`token_classification.py::preprocess`）

- 读取 `data/raw/data.jsonl`，将字符串标签经 `ClassLabel.str2int` 转为 id；
- 按 8:1:1 切分 train / valid / test；
- **子词对齐**：tokenizer 以 `is_split_into_words=True` 按字切分，再通过 `word_ids(batch_index=i)`
  把每个 token 映射回原字下标；`[CLS]` / `[SEP]` 以及子词引入的位置标注为 `-100`，
  使 `CrossEntropyLoss` 忽略这些位置；
- 处理结果以 `datasets.save_to_disk` 落盘到 `data/processed/{train,valid,test}`（已存在则跳过）。

### 2. 模型微调（`token_classification.py::train`）

- `num_train_epochs=20`、`lr=2e-5`、`cosine` 调度、`warmup_ratio=0.2`、`weight_decay=0.01`；
- 根据硬件自动选择 `bf16` / `fp16`（`torch.cuda.is_bf16_supported()`）；
- 每 500 步评估并保存，`save_total_limit=3`、`save_only_model=True`；
- 指标：`accuracy` / `precision` / `recall` / `f1`（macro，`zero_division=0`），
  以 `eval_f1` 选择最优模型；
- 训练结束后对比**最终评估分数**与**历史最优**：更优则直接保存到 `finetuned/best`，
  否则回退复制 `best_model_checkpoint`，最后在 test 集上输出 `test_*` 指标。

### 3. 推理（`token_classification.py::predict`）

- 输入按字拆分为 `list(text)`，保证与训练时的切分粒度一致；
- 批量推理，`argmax` 取标签 id，`word_ids` 过滤掉 `[CLS]` / `[SEP]`；
- 原字符串中的空格补 `O` 标签，**保证预测序列与原始字符串逐字对齐**。

### 4. 地址提取（`address_alignment.py::address_extract`）

扫描标签序列，把连续同类标签合并为文本片段，落入 `prov/city/district/town/detail/name/phone`。

### 5. 地址校验与对齐（`address_alignment.py::address_check`）

1. **电话校验**：11 位手机号（首位为 1）或座机号正则；不合法置 `None`；
2. 省市区镇全为空则直接返回；
3. **容错查询**：除按级别正常查询外，额外构造跨级兜底条件，处理模型把「省/市」标错位的情况；
4. 用 `region_type=%s and name like %s` 条件以 `OR` 拼接，查询标准地址库的 `full_name`；
5. 将 `full_name` 按空格切分后构造成**地址树**（`dict` 嵌套，叶子为路段 `list`），再展平为地址链；
6. 地址链拼接为文本后，用 **RapidFuzz `fuzz.ratio`** 与原始文本计算相似度，取分数最高的作为规范化结果；
7. 候选为空时返回全空结果（避免 `max()` 抛 `ValueError`）。

---

## 项目结构

```
address_alignment/
├── data/
│   ├── raw/data.jsonl                 # 原始标注数据
│   ├── processed/{train,valid,test}/  # 预处理后的 HF Dataset
│   └── region.sql                     # 省市区标准地址库（导入 MySQL）
├── pretrained/
│   └── roberta-small-wwm-chinese-cluecorpussmall/   # 本地预训练模型（约 92 MB）
├── logs/                              # TensorBoard 训练日志
├── templates/index.html               # 简单前端页面
├── token_classification.py            # 数据预处理 + 训练 + 推理
├── address_alignment.py               # 地址提取 + 地址库校验对齐
├── sql_import.py                      # 地址库导入 MySQL
├── app.py                             # FastAPI 服务入口
├── config.py                          # 路径 / 设备 / 标签 / 数据库配置
├── requirements.txt
└── README.md
```

> 训练产物 `finetuned/`（微调权重）不入库，需本地训练生成。

---

## 快速开始

### 1. 安装依赖

```bash
git clone https://github.com/chenjack-oss/address_alignment.git
cd address_alignment
pip install -r requirements.txt
```

### 2. 准备地址库

确保本地 MySQL 可用（需要系统存在 `mysql` 命令行客户端），然后按需通过环境变量覆盖连接配置：

```bash
export MYSQL_HOST=127.0.0.1
export MYSQL_PORT=3306
export MYSQL_USER=root
export MYSQL_PASSWORD=abc123456
export MYSQL_DATABASE=region

python sql_import.py     # 建库 + 导入 data/region.sql
```

> 未设置环境变量时，回退使用 `config.MYSQL_CONFIG` 中的默认值。

### 3. 训练模型

```bash
python token_classification.py
```

执行 `train(config.ROBERTA_SMALL)` 后，模型保存到 `finetuned/best`，日志写入 `logs/`。
脚本末尾还会跑一组样例预测，用于快速验证效果。

### 4. 启动服务

```bash
python app.py            # 等价于 uvicorn.run(app, host="0.0.0.0", port=8089)
```

浏览器访问 <http://127.0.0.1:8089/> 使用页面，或直接调接口：

```bash
curl -X POST http://127.0.0.1:8089/address_alignment \
  -H "Content-Type: application/json" \
  -d '{"message": "中国浙江省杭州市余杭区葛墩路27号楼傅婷15830444519"}'
```

响应示例：

```json
{
  "prov": "浙江省",
  "city": "杭州市",
  "district": "余杭区",
  "town": null,
  "detail": "葛墩路27号楼",
  "name": "傅婷",
  "phone": "15830444519"
}
```

---

## 已知限制

- 微调权重未入库，需先执行训练；`app.py` / `address_alignment.py` 依赖 `finetuned/best` 存在。
- 地址对齐强依赖 MySQL 中 `region` 表的完整性，地址库缺失会导致该级别字段为空。
- 编辑距离对长地址的区分度有限，后续可替换为向量检索或向量 + 编辑距离混合打分。
