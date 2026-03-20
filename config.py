import torch
from pathlib import Path

MYSQL_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "abc123456",
    "database": "region",
    "charset": "utf8mb4",
}

# SQl文件路径
SQL_FILE_PATH = Path("data/region.sql")
# 原始数据路径
RAW_DATA_PATH = Path("data/raw")
# 已处理数据存放路径
PROCESSED_DATA_PATH = Path("data/processed")
# 模型参数保存路径
FINETUNED_PATH = Path("finetuned")
# TensorBoard 日志保存路径
LOGS_PATH = Path("logs")
# 本地预训练模型路径
PRETRAINED_PATH = Path(r"D:\PycharmProjects\student_projects\0716_llm\address_alignment\pretrained")
ROBERTA_SMALL = PRETRAINED_PATH / "roberta-small-wwm-chinese-cluecorpussmall"

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 标签
LABELS = [
    "O",
    "B-prov",
    "I-prov",
    "E-prov",
    "B-city",
    "I-city",
    "E-city",
    "B-district",
    "I-district",
    "E-district",
    "S-district",
    "B-town",
    "I-town",
    "E-town",
    "S-town",
    "B-detail",
    "I-detail",
    "E-detail",
    "S-detail",
    "B-name",
    "I-name",
    "E-name",
    "B-phone",
    "I-phone",
    "E-phone",
]
