# 修复OpenPR在数据库里的指标命名问题

# filename: fix_db.py
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database.db"

def fix_typo():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 检查是否存在需要修改的数据
    cursor.execute("SELECT count(*) FROM metrics WHERE metric_type = 'openPR'")
    count = cursor.fetchone()[0]
    
    if count > 0:
        print(f"发现 {count} 条 'openPR' 数据，正在更新为 'OpenPR'...")
        cursor.execute("UPDATE metrics SET metric_type = 'OpenPR' WHERE metric_type = 'openPR'")
        conn.commit()
        print("更新完成。")
    else:
        print("未发现 'openPR' 数据，可能已经修正或数据不存在。")
        
    conn.close()

if __name__ == "__main__":
    fix_typo()