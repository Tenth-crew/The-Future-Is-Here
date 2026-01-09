# filename: init_db.py
import os
import json
import pandas as pd
import sqlite3
from pathlib import Path

# 配置路径
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "database.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. 创建表结构
    # 仓库元数据表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS repos (
            repo_name TEXT PRIMARY KEY,
            score REAL,
            description TEXT
        )
    ''')

    # 指标数据表 (存储所有的折线图点位)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo_name TEXT,
            metric_type TEXT, -- mergePR, IssueComment 等
            data_category TEXT, -- history, actual, prediction
            date TEXT,
            value REAL,
            FOREIGN KEY(repo_name) REFERENCES repos(repo_name)
        )
    ''')
    
    conn.commit()
    print("表结构创建完成。")

    # 2. 加载 CSV 数据
    csv_path = DATA_DIR / "processed_repo_data.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # 假设CSV列名为 Repo_Name, score, score_description
        records = df[['Repo_Name', 'score', 'score_description']].to_dict('records')
        
        for row in records:
            cursor.execute(
                "INSERT OR REPLACE INTO repos (repo_name, score, description) VALUES (?, ?, ?)",
                (row['Repo_Name'], row['score'], row['score_description'])
            )
        print(f"导入了 {len(records)} 个仓库的基本信息。")
    
    # 3. 加载 JSON 指标数据
    echart_dir = DATA_DIR / "echart_data"
    if echart_dir.exists():
        # 遍历每个仓库文件夹
        for repo_folder in os.listdir(echart_dir):
            repo_path = echart_dir / repo_folder
            if repo_path.is_dir():
                # 遍历该仓库下的所有JSON文件
                for json_file in os.listdir(repo_path):
                    if json_file.endswith(".json"):
                        # 解析文件名获取指标类型 (假设文件名如 IssueComment_data.json)
                        metric_name = json_file.replace("_data.json", "")
                        
                        with open(repo_path / json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                        # 插入数据库
                        repo_name = data.get('repo_name', repo_folder) # 优先用json里的，否则用文件夹名
                        
                        # 处理 history, actual, prediction 三个数组
                        for category in ['history', 'actual', 'prediction']:
                            if category in data:
                                for point in data[category]:
                                    cursor.execute(
                                        "INSERT INTO metrics (repo_name, metric_type, data_category, date, value) VALUES (?, ?, ?, ?, ?)",
                                        (repo_name, metric_name, category, point['date'], point['value'])
                                    )
        print("指标数据导入完成。")
    
    conn.commit()
    conn.close()
    print("数据库初始化完毕。")

if __name__ == "__main__":
    init_db()