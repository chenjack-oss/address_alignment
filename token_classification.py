import os
import torch
import config
import shutil
import numpy as np
from pathlib import Path
from torch.cuda import is_bf16_supported
from sklearn.metrics import precision_recall_fscore_support
from datasets import load_dataset, ClassLabel, load_from_disk
from transformers import (
    Trainer,
    EvalPrediction,
    TrainingArguments,
    BertTokenizerFast,
    BertForTokenClassification,
    DataCollatorForTokenClassification,
)


# 数据预处理
def preprocess(
    src_path: Path,
    tgt_path: Path,
    tokenizer: BertTokenizerFast,
    max_input_len: int,
    label_names: list[str],
    text_col="text",
    train_size=0.8,
    test_size=0.1,
):
    """处理分类任务数据集"""

    if os.path.exists(tgt_path):
        return
    # 读取文件，类别 → int
    label_feature = ClassLabel(names=label_names)
    datas = load_dataset("json", data_files=str(src_path))["train"].map(
        lambda x: {"labels": [label_feature.str2int(label) for label in x["labels"]]}
    )
    # 随机打乱并切分
    datas = datas.train_test_split(test_size=test_size, shuffle=True)
    train_val = datas["train"].train_test_split(train_size=train_size / (1 - test_size))
    datas["train"], datas["valid"] = train_val["train"], train_val["test"]

    def _map_fn(examples):
        # 分词之后 text 被添加了 [CLS] 和 [SEP]，需要将 label 与分词后的 text 对齐
        tokenized = tokenizer(
            examples[text_col],
            truncation=True,
            max_length=max_input_len,
            is_split_into_words=True,
        )
        # label 中与 [CLS] 和 [SEP] 对应的位置设置为 -100


        labels = [
            [
                label_seq[j] if j is not None else -100
                for j in tokenized.word_ids(batch_index=i)
            ]
            for i, label_seq in enumerate(examples["labels"])
        ]
        tokenized["labels"] = labels
        return tokenized

    # 分词
    datas = {
        k: v.map(
            _map_fn,
            batched=True,
            remove_columns=[c for c in v.column_names if c != "labels"],
        )
        for k, v in datas.items()
    }
    # 保存数据集
    for type in ["train", "valid", "test"]:
        datas[type].save_to_disk(tgt_path / type)


# 模型训练
def train(model_path: Path):
    tokenizer = BertTokenizerFast.from_pretrained(model_path)

    # 加载数据
    preprocess(
        src_path=config.RAW_DATA_PATH / "data.jsonl",
        tgt_path=config.PROCESSED_DATA_PATH,
        tokenizer=tokenizer,
        max_input_len=128,
        label_names=config.LABELS,
    )
    train_dataset = load_from_disk(config.PROCESSED_DATA_PATH / "train")
    valid_dataset = load_from_disk(config.PROCESSED_DATA_PATH / "valid")
    test_dataset = load_from_disk(config.PROCESSED_DATA_PATH / "test")

    # 加载模型
    model = BertForTokenClassification.from_pretrained(
        model_path, num_labels=len(config.LABELS)
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir=config.FINETUNED_PATH,  # 保存路径
        num_train_epochs=20,  # 训练轮次
        per_device_train_batch_size=64,  # 训练批次
        per_device_eval_batch_size=64,  # 验证批次
        learning_rate=2e-5,  # 学习率
        warmup_ratio=0.2,  # 预热比例
        lr_scheduler_type="cosine",  # 学习率调度器
        weight_decay=0.01,  # 权重衰减
        bf16=is_bf16_supported(),  # 是否使用bf16
        fp16=not is_bf16_supported(),  # 是否使用fp16
        eval_strategy="steps",  # 验证策略
        save_strategy="steps",  # 保存策略
        eval_steps=500,  # 验证步数
        save_steps=500,  # 保存步数
        logging_steps=200,  # 日志记录步数
        logging_dir=config.LOGS_PATH,  # 日志保存路径
        save_total_limit=3,  # 模型保存数量限制
        save_only_model=True,  # 只保存模型
        metric_for_best_model="eval_f1",  # 用于评估最优模型的指标
        greater_is_better=True,  # 指标值越大越好
        disable_tqdm=False,  # 是否禁用进度条
        logging_first_step=True,  # 是否在第一步记录日志
    )

    # 评估指标
    def compute_metrics(eval_pred: EvalPrediction):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        mask = labels != -100
        y_true = labels[mask]
        y_pred = preds[mask]
        acc = (y_true == y_pred).mean()
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        outputs_result ={
            "accuracy": float(acc),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
        }
        print(outputs_result)
        return outputs_result

    # 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    # 训练
    trainer.train()

    # 最终验证
    final_eval_metrics = trainer.evaluate()
    print(final_eval_metrics)
    # 获取当前最佳模型的f1分数
    best_metric = getattr(trainer.state, "best_metric", 0.0)
    best_dir = config.FINETUNED_PATH / "best"
    # 如果最终验证的f1分数高于之前的最佳分数，则保存为最佳模型
    if final_eval_metrics.get("eval_f1", 0.0) >= best_metric:
        trainer.save_model(best_dir)
        tokenizer.save_pretrained(best_dir)
    # 如果最终模型不是最佳的，则将最佳检查点保存为 best
    else:
        best_checkpoint = trainer.state.best_model_checkpoint
        if best_checkpoint:
            if os.path.exists(best_dir):
                shutil.rmtree(best_dir)
            shutil.copytree(best_checkpoint, best_dir, dirs_exist_ok=True)
            tokenizer.save_pretrained(best_dir)

    # 测试
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    print(test_metrics)


@torch.inference_mode()
def predict(
    text: str | list[str],
    model: BertForTokenClassification,
    tokenizer: BertTokenizerFast,
    label_names: list[str],
    batch_size=64,
):
    model.eval()
    # 获取模型所在设备
    device = next(model.parameters()).device
    # 统一转换为列表
    texts = text if isinstance(text, list) else [text]
    # 逐批次处理
    res: list[list[str]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # 将每个字符串拆成字列表：确保 tokenizer 能按字对齐 word_ids
        batch_words = [list(t.replace(" ", "")) for t in batch]
        # 分词
        tokenized = tokenizer(
            batch_words,
            max_length=128,
            truncation=True,
            padding=True,
            return_tensors="pt",
            is_split_into_words=True,
        ).to(device)
        outputs = model(tokenized["input_ids"], tokenized["attention_mask"])
        preds = torch.argmax(outputs["logits"], dim=-1).detach().cpu()
        # 将 preds 中对 [CLS] 和 [SEP] 的预测值过滤掉
        filtered_preds = [
            [
                label_names[label_seq[j + 1].item()]
                for j in tokenized.word_ids(batch_index=i)
                if j is not None
            ]
            for i, label_seq in enumerate(preds)
        ]
        # 将输入中的空格标注上 'O'
        filled_preds = []
        for string, labels in zip(batch, filtered_preds):
            labels_iter = iter(labels)
            filled_preds.append(
                [next(labels_iter) if char != " " else "O" for char in string]
            )
        res.extend(filled_preds)
    return res if isinstance(text, list) else res[0]



if __name__ == "__main__":
    train(config.ROBERTA_SMALL)

    model_path = config.FINETUNED_PATH / "best"
    model = BertForTokenClassification.from_pretrained(model_path).to(config.DEVICE)
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    texts = [
        "中国浙江省杭州市余杭区葛墩路27号楼傅婷15830444519",
        "北京市市辖区通州区永乐店镇27号楼汪明13334219987",
        "高霞 13139243427北京市市辖区东风街道27号楼",
        "新疆维吾尔自治区划阿拉尔市金杨镇27号楼 刘燕 14727827196",
        "甘肃省南市文县碧口镇27号楼陈桂兰 13939269190",
        "陕西省渭南市华阴市罗镇27号楼 赵鑫15687584092",
        "邱金凤18582166250西藏自治区拉萨市墨竹工卡县工卡镇27号楼",
        "广州市花都区花东镇27号楼张荣   18736672007",
    ]
    res = predict(texts, model, tokenizer, config.LABELS)
    for ts, rs in zip(texts, res):
        for t, r in zip(ts, rs):
            print(t, r)
        print()
