---
language: zh
datasets: CLUECorpusSmall
widget: 
- text: "北京是[MASK]国的首都。"



---
# Chinese Whole Word Masking RoBERTa Miniatures

## Model description

This is the set of 6 Chinese Whole Word Masking RoBERTa models pre-trained by [UER-py](https://github.com/dbiir/UER-py/), which is introduced in [this paper](https://arxiv.org/abs/1909.05658). Besides, the models could also be pre-trained by [TencentPretrain](https://github.com/Tencent/TencentPretrain) introduced in [this paper](https://arxiv.org/abs/2212.06385), which inherits UER-py to support models with parameters above one billion, and extends it to a multimodal pre-training framework.

[Turc et al.](https://arxiv.org/abs/1908.08962) have shown that the standard BERT recipe is effective on a wide range of model sizes. Following their paper, we released the 6 Chinese Whole Word Masking RoBERTa models. In order to facilitate users in reproducing the results, we used a publicly available corpus and word segmentation tool, and provided all training details.

You can download the 6 Chinese RoBERTa miniatures either from the [UER-py Modelzoo page](https://github.com/dbiir/UER-py/wiki/Modelzoo), or via HuggingFace from the links below:

|          |           Link           |
| -------- | :-----------------------: |
| **Tiny**  | [**2/128 (Tiny)**][2_128] |
| **Mini**  | [**4/256 (Mini)**][4_256] |
| **Small**  | [**4/512 (Small)**][4_512] |
| **Medium**  | [**8/512 (Medium)**][8_512] |
| **Base** | [**12/768 (Base)**][12_768] |
| **Large** | [**24/1024 (Large)**][24_1024] |

Here are scores on the devlopment set of six Chinese tasks:

| Model              | Score | book_review | chnsenticorp | lcqmc | tnews(CLUE) | iflytek(CLUE) | ocnli(CLUE) |
| ------------------ | :---: | :----: | :----------: | :---: | :---------: | :-----------: | :---------: |
| RoBERTa-Tiny-WWM   | 72.2  |  83.7  |     91.8     | 81.8  |    62.1     |     55.4      |    58.6     |
| RoBERTa-Mini-WWM   | 76.3  |  86.4  |     93.0     | 86.8  |    64.4     |     58.7      |    68.8     |
| RoBERTa-Small-WWM  | 77.6  |  88.1  |     93.8     | 87.2  |    65.2     |     59.6      |    71.4     |
| RoBERTa-Medium-WWM | 78.6  |  89.3  |     94.4     | 88.8  |    66.0     |     59.9      |    73.2     |
| RoBERTa-Base-WWM   | 80.2  |  90.6  |     95.8     | 89.4  |    67.5     |     61.8      |    76.2     |
| RoBERTa-Large-WWM  | 81.1  |  91.1  |     95.8     | 90.0  |    68.5     |     62.1      |    79.1     |

For each task, we selected the best fine-tuning hyperparameters from the lists below, and trained with the sequence length of 128:

- epochs: 3, 5, 8
- batch sizes: 32, 64
- learning rates: 3e-5, 1e-4, 3e-4

## How to use

You can use this model directly with a pipeline for masked language modeling:

```python
>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='uer/roberta-tiny-wwm-chinese-cluecorpussmall')
>>> unmasker("北京是[MASK]国的首都。")
[
    {'score': 0.294228732585907, 
     'token': 704, 
     'token_str': '中', 
     'sequence': '北 京 是 中 国 的 首 都 。'},
    {'score': 0.19691626727581024, 
     'token': 1266, 
     'token_str': '北', 
     'sequence': '北 京 是 北 国 的 首 都 。'},
    {'score': 0.1070084273815155, 
     'token': 7506, 
     'token_str': '韩', 
     'sequence': '北 京 是 韩 国 的 首 都 。'},
    {'score': 0.031527262181043625, 
     'token': 2769, 
     'token_str': '我', 
     'sequence': '北 京 是 我 国 的 首 都 。'},
    {'score': 0.023054633289575577, 
     'token': 1298, 
     'token_str': '南', 
     'sequence': '北 京 是 南 国 的 首 都 。'}
]
    


```

Here is how to use this model to get the features of a given text in PyTorch:

```python
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('uer/roberta-base-wwm-chinese-cluecorpussmall')
model = BertModel.from_pretrained("uer/roberta-base-wwm-chinese-cluecorpussmall")
text = "用你喜欢的任何文本替换我。"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```

and in TensorFlow:

```python
from transformers import BertTokenizer, TFBertModel
tokenizer = BertTokenizer.from_pretrained('uer/roberta-base-wwm-chinese-cluecorpussmall')
model = TFBertModel.from_pretrained("uer/roberta-base-wwm-chinese-cluecorpussmall")
text = "用你喜欢的任何文本替换我。"
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
```

## Training data

[CLUECorpusSmall](https://github.com/CLUEbenchmark/CLUECorpus2020/) is used as training data. 

## Training procedure

Models are pre-trained by [UER-py](https://github.com/dbiir/UER-py/) on [Tencent Cloud](https://cloud.tencent.com/). We pre-train 1,000,000 steps with a sequence length of 128 and then pre-train 250,000 additional steps with a sequence length of 512. We use the same hyper-parameters on different model sizes.

[jieba](https://github.com/fxsjy/jieba) is used as word segmentation tool.

Taking the case of Whole Word Masking RoBERTa-Medium

Stage1:

```
python3 preprocess.py --corpus_path corpora/cluecorpussmall.txt \
                      --vocab_path models/google_zh_vocab.txt \
                      --dataset_path cluecorpussmall_seq128_dataset.pt \
                      --processes_num 32 --seq_length 128 \
                      --dynamic_masking --data_processor mlm
```

```
python3 pretrain.py --dataset_path cluecorpussmall_seq128_dataset.pt \
                    --vocab_path models/google_zh_vocab.txt \
                    --config_path models/bert/medium_config.json \
                    --output_model_path models/cluecorpussmall_wwm_roberta_medium_seq128_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 1000000 --save_checkpoint_steps 100000 --report_steps 50000 \
                    --learning_rate 1e-4 --batch_size 64 \
                    --whole_word_masking \
                    --data_processor mlm --target mlm
```

Stage2:

```
python3 preprocess.py --corpus_path corpora/cluecorpussmall.txt \
                      --vocab_path models/google_zh_vocab.txt \
                      --dataset_path cluecorpussmall_seq512_dataset.pt \
                      --processes_num 32 --seq_length 512 \
                      --dynamic_masking --data_processor mlm
```

```
python3 pretrain.py --dataset_path cluecorpussmall_seq512_dataset.pt \
                    --vocab_path models/google_zh_vocab.txt \
                    --pretrained_model_path models/cluecorpussmall_wwm_roberta_medium_seq128_model.bin-1000000 \
                    --config_path models/bert/medium_config.json \
                    --output_model_path models/cluecorpussmall_wwm_roberta_medium_seq512_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 250000 --save_checkpoint_steps 50000 --report_steps 10000 \
                    --learning_rate 5e-5 --batch_size 16 \
                    --whole_word_masking \
                    --data_processor mlm --target mlm
```

Finally, we convert the pre-trained model into Huggingface's format:

```
python3 scripts/convert_bert_from_uer_to_huggingface.py --input_model_path models/cluecorpussmall_wwm_roberta_medium_seq512_model.bin-250000 \
                                                        --output_model_path pytorch_model.bin \
                                                        --layers_num 8 --type mlm
```

### BibTeX entry and citation info

```
@article{zhao2019uer,
  title={UER: An Open-Source Toolkit for Pre-training Models},
  author={Zhao, Zhe and Chen, Hui and Zhang, Jinbin and Zhao, Xin and Liu, Tao and Lu, Wei and Chen, Xi and Deng, Haotang and Ju, Qi and Du, Xiaoyong},
  journal={EMNLP-IJCNLP 2019},
  pages={241},
  year={2019}
}

@article{zhao2023tencentpretrain,
  title={TencentPretrain: A Scalable and Flexible Toolkit for Pre-training Models of Different Modalities},
  author={Zhao, Zhe and Li, Yudong and Hou, Cheng and Zhao, Jing and others},
  journal={ACL 2023},
  pages={217},
  year={2023}
```

[2_128]:https://huggingface.co/uer/roberta-tiny-wwm-chinese-cluecorpussmall
[4_256]:https://huggingface.co/uer/roberta-mini-wwm-chinese-cluecorpussmall
[4_512]:https://huggingface.co/uer/roberta-small-wwm-chinese-cluecorpussmall
[8_512]:https://huggingface.co/uer/roberta-medium-wwm-chinese-cluecorpussmall
[12_768]:https://huggingface.co/uer/roberta-base-wwm-chinese-cluecorpussmall
[24_1024]:https://huggingface.co/uer/roberta-large-wwm-chinese-cluecorpussmall