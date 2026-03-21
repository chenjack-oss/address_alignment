# 地址抽取与对齐项目 (Address Alignment)

## 项目简介

本项目是一个基于**预训练语言模型（BERT）** 的地址实体识别与对齐工具。  
主要功能：从非结构化的中文地址文本中，自动提取出**省、市、区、镇、详细地址、姓名、电话**等结构化信息，并利用数据库中的标准地址库进行**编辑距离校验与纠错**，最终输出规范化的地址组件。

项目来源于尚硅谷大模型技术实战课程，整合了自然语言处理（序列标注）、数据预处理、模型微调、地址库匹配等完整流程。

---

## 技术栈

- **Python 3.8+**
- **PyTorch** / **Transformers** （HuggingFace）
- **BERT** 中文预训练模型（`bert-base-chinese`）
- **MySQL**（地址库存储与查询）
- **RapidFuzz**（编辑距离相似度计算）
- **PyMySQL**（数据库连接）

---

## 核心流程

1. **数据预处理**
   - 对原始地址文本进行 tokenization，并保证 token 与 label 对齐（处理特殊 token 和子词分词）
   - 动态填充（DataCollatorForTokenClassification），避免固定长度带来的资源浪费

2. **模型微调**
   - 使用 `BertForTokenClassification` 进行序列标注训练
   - 配置 TrainingArguments 与 Trainer，支持 bf16/fp16、学习率预热、早停等策略
   - 评估指标：F1 分数

3. **地址对齐模块**
   - 模型预测 → 提取各级别地址片段
   - 电话号码正则校验
   - 基于 MySQL 地址库的模糊匹配：将模型预测的省市区镇与数据库中的 `full_name` 进行模糊查询，构建可能地址树，通过编辑距离（RapidFuzz）选出最匹配的标准地址

---

## 项目结构
address_alignment/
├── data/ # 原始数据与处理后的数据集
├── finetuned/ # 微调后的模型权重
├── logs/ # 训练日志
├── pretrained/ # 预训练模型（BERT）
├── templates/ # 模板文件（如有）
├── token_classification/ # 序列标注相关代码
│ ├── init.py
│ ├── predict.py # 预测接口
│ └── train.py # 训练脚本
├── sql_import/ # MySQL 地址库导入脚本
├── app.py # 主入口（地址对齐接口）
├── config.py # 配置文件（路径、超参数、数据库配置等）
├── requirements.txt # 依赖列表
└── README.md # 本文件


---

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/chenjack-oss/address_alignment.git
cd address_alignment
