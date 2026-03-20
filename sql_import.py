import config
import subprocess

tag = {
    "error": "\033[1;31m[ERROR]\033[0m",
    "success": "\033[1;32m[SUCCESS]\033[0m",
    "processing": "\033[1;34m[PROCESSING]\033[0m",
}


# 导入数据到 MySQL
def import_db(mysql_config, sql_file_path):
    """使用 sql 文件建库并导入数据"""

    mysql_cmd_prefix = [
        "mysql",
        f"-h{mysql_config['host']}",
        f"-P{mysql_config['port']}",
        f"-u{mysql_config['user']}",
        f"-p{mysql_config['password']}",
        f"--default-character-set={mysql_config['charset']}",
    ]

    # 建库
    print(f"{tag['processing']} 创建 {mysql_config['database']}")
    cmd = mysql_cmd_prefix + [
        "-e",
        f"create database if not exists {mysql_config['database']};",
    ]
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"{tag['error']} {result.stderr.splitlines()[1:][0]}")
        return
    print(f"{tag['success']} {mysql_config['database']} 创建成功")

    # 导入数据
    print(f"{tag['processing']} 导入表单数据")
    with open(sql_file_path, "r") as sql_file:
        result = subprocess.run(
            mysql_cmd_prefix + [mysql_config["database"]],
            stdin=sql_file,
            stderr=subprocess.PIPE,
            text=True,
        )
    if result.returncode != 0:
        print(f"{tag['error']} {result.stderr.splitlines()[1:][0]}")
        return
    print(f"{tag['success']} 表单数据导入成功")


import_db(config.MYSQL_CONFIG, config.SQL_FILE_PATH)
