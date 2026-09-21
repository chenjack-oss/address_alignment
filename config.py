import os
import torch
from pathlib import Path

# 项目根目录（本文件所在目录），所有相对路径均以此为基准，
# 避免脚本在不同工作目录下执行时找不到数据/模型。
BASE_DIR = Path(__file__).resolve().parent

# 数据库连接配置，支持通过环境变量覆盖，便于在不同环境中部署。
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", "abc123456"),
    "database": os.getenv("MYSQL_DATABASE", "region"),
    "charset": "utf8mb4",
}

# SQl文件路径
SQL_FILE_PATH = BASE_DIR / "data/region.sql"
# 原始数据路径
RAW_DATA_PATH = BASE_DIR / "data/raw"
# 已处理数据存放路径
PROCESSED_DATA_PATH = BASE_DIR / "data/processed"
# 模型参数保存路径
FINETUNED_PATH = BASE_DIR / "finetuned"
# TensorBoard 日志保存路径
LOGS_PATH = BASE_DIR / "logs"
# 本地预训练模型路径
PRETRAINED_PATH = BASE_DIR / "pretrained"
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
