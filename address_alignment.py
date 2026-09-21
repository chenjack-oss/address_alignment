import re
from typing import Optional
import config
import pymysql
from pymysql import cursors
from rapidfuzz import fuzz
from token_classification import predict
from transformers import BertForTokenClassification, BertTokenizerFast


def address_alignment(text: str, model, tokenizer, label_name, mysql_config) -> dict:
    """地址对齐"""
    # 序列标注
    tagging = predict(text, model, tokenizer, label_name)
    # 提取各级别地址
    address = address_extract(text, tagging)
    # 校验地址信息
    correct_address = address_check(text, address, mysql_config)
    address.update(correct_address)
    return address


def address_extract(text: str, tagging: list[str]) -> dict[str, Optional[str]]:
    """地址提取"""
    tagging = [i[2:] for i in tagging]
    # 各级别对应地址信息
    address:dict[str,Optional[str]] = {
        "prov": None,
        "city": None,
        "district": None,
        "town": None,
        "detail": None,
        "name": None,
        "phone": None,
    }
    # 提取出各级别地址信息
    start_pos = 0
    tag_len = len(tagging)
    for end_pos in range(tag_len):
        # 如果到结尾、或 end_pos 的下一个位置不是同一类
        if (end_pos == tag_len - 1) or (tagging[end_pos + 1] != tagging[start_pos]):
            if tagging[start_pos] != "":
                address[tagging[start_pos]] = text[start_pos : end_pos + 1]
            start_pos = end_pos + 1
    return address


def address_check(text: str, address: dict[str, Optional[str]], mysql_config) -> dict[str, Optional[str]]:
    """地址校验"""

    # 检查电话号码
    if address["phone"]:
        # 检查是否为手机号码
        if not (
            len(address["phone"]) == 11
            and address["phone"].isdigit()
            and address["phone"][0] == "1"
        ):
            # 检查是否位座机号码
            pattern = r"^(?:\+?\d{1,4}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{6,11}$"
            if not re.match(pattern, address["phone"]):
                address["phone"] = None

    # 如果地址部分为空则结束
    if all(address[k] is None for k in ["prov", "city", "district", "town"]):
        return address

    def flatten_address_tree(tree, chain=None):
        """将树展平"""
        if chain is None:
            chain = []
        chains = []
        # 非叶节点
        if isinstance(tree, dict) and tree:
            for k, v in tree.items():
                chains.extend(flatten_address_tree(v, chain + [k]))
        # 叶节点
        elif isinstance(tree, list) and tree:
            for road in tree:
                chains.append(chain + [road])
        else:
            chains.append(chain)
        return chains

    region_type_ids = [2, 3, 4, 5]
    region_type_names = ["prov", "city", "district", "town"]
    with pymysql.connect(**mysql_config) as mysql_conn:
        with mysql_conn.cursor(pymysql.cursors.DictCursor) as cursor:
            params_list = [
                (i, address[k]) for i, k in zip(region_type_ids, region_type_names)
            ]
            # 处理 省 市 区 标签错位
            params_list.extend(
                [
                    (2, address["city"]),
                    (3, address["prov"]),
                    (3, address["district"]),
                ]
            )
            # 过滤空值
            params_list = [(i[0], f"%{i[1]}%") for i in params_list if i[1]]

            # 将多个(region_type=%s, name like %s)条件用or拼接,查询每个层级对应的full_name
            placeholders = "(region_type=%s and name like %s)"
            sql = (
                f"select full_name from region where {placeholders}"
                + f" or {placeholders}" * (len(params_list) - 1)
            )
            params = [i for pair in params_list for i in pair]
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            prefixes = [row["full_name"].split(" ") for row in rows]

    # 合并前缀
    # {
    #     "prov": {
    #         "city": {
    #             "dist": ["road"],
    #         },
    #     },
    # }
    # 将所有可能的地址，构造成一个dict，后面将其打平
    address_tree = {}
    leaf_id = region_type_ids[-1] - region_type_ids[0]
    for prefix in prefixes:
        node = address_tree
        for i in range(len(prefix)):
            if i == leaf_id:
                node.append(prefix[i])
            elif i == leaf_id - 1:
                node = node.setdefault(prefix[i], [])
            else:
                node = node.setdefault(prefix[i], {})

    # 展平
    # [["prov", "city", "dist", "road"]]
    address_chains = flatten_address_tree(address_tree)

    # 转换为地址文本
    address_texts = ["".join(i) for i in address_chains]

    # 基于编辑距离计算两字符串的相似度
    # 编辑距离用来衡量两个字符串之间相似度
    # 表示从一个字符串A变成另一个字符串B，所需要的最少单字符编辑操作次数
    # 编辑操作包括插入、删除、替换
    scores = [fuzz.ratio(i, text) for i in address_texts]

    # 取分数最高的结果
    correct_address = {k: None for k in region_type_names}
    # 地址库中没有匹配到候选地址时，直接返回全空结果，避免 max() 抛 ValueError
    if not scores:
        return correct_address
    correct_address_chain = address_chains[scores.index(max(scores))]
    correct_address.update(zip(region_type_names, correct_address_chain))

    return correct_address


if __name__ == "__main__":
    model_path = config.FINETUNED_PATH / "best"
    model = BertForTokenClassification.from_pretrained(model_path).to(config.DEVICE)
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    texts = [
        # "中国浙江省杭州市余杭区葛墩路27号楼傅婷15830444519",
        # "北京市市辖区通州区永乐店镇27号楼汪明13334219987",
        # "高霞 13139243427北京市市辖区东风街道27号楼",
        # "新疆维吾尔自治区划阿拉尔市金杨镇27号楼 刘燕 14727827196",
        # "甘肃省南市文县碧口镇27号楼陈桂兰 13939269190",
        # "陕西省渭南市华阴市罗镇27号楼 赵鑫15687584092",
        # "邱金凤18582166250西藏自治区拉萨市墨竹工卡县工卡镇27号楼",
        "北京市拱辰街道27号楼2单元陈桂兰13939269190",
        # ""
        "河北省朔州市玉田县林北仓镇27号楼5单元张荣18736672007",
        "河北省朔州市玉田县林南仓镇27号楼5单元张荣18736672007"
    ]
    for text in texts:
        address_dict = address_alignment(
            text, model, tokenizer, config.LABELS, config.MYSQL_CONFIG
        )
        print(address_dict)
