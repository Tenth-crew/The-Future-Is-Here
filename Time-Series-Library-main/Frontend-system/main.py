# filename: main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import sqlite3
import uvicorn
from pathlib import Path

app = FastAPI()

# 挂载静态文件和模板
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
DB_PATH = BASE_DIR / "database.db"

# 数据库连接辅助函数
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# 路由 1: 首页
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    conn = get_db_connection()
    # 获取分数前5的仓库
    top_repos = conn.execute("SELECT * FROM repos ORDER BY score DESC LIMIT 5").fetchall()
    # 获取所有仓库用于轮播 (限制数量避免页面过大)
    all_repos = conn.execute("SELECT * FROM repos").fetchall()
    conn.close()
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "top_repos": [dict(row) for row in top_repos],
        "all_repos": [dict(row) for row in all_repos]
    })

# 路由 2: 详情页
@app.get("/repo/{repo_name}", response_class=HTMLResponse)
async def read_repo(request: Request, repo_name: str):
    conn = get_db_connection()
    repo = conn.execute("SELECT * FROM repos WHERE repo_name = ?", (repo_name,)).fetchone()
    
    if not repo:
        conn.close()
        raise HTTPException(status_code=404, detail="Repo not found")
        
    conn.close()
    return templates.TemplateResponse("details.html", {
        "request": request,
        "repo": dict(repo)
    })

# API: 获取指定仓库的指标数据 (供ECharts异步加载)
@app.get("/api/metrics/{repo_name}/{metric_type}")
async def get_metrics(repo_name: str, metric_type: str):
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT data_category, date, value 
        FROM metrics 
        WHERE repo_name = ? AND metric_type = ?
        ORDER BY date ASC
    """, (repo_name, metric_type)).fetchall()
    conn.close()
    
    # 格式化数据给前端
    result = {"history": [], "actual": [], "prediction": []}
    for row in rows:
        cat = row['data_category']
        if cat in result:
            result[cat].append({"date": row['date'], "value": row['value']})
            
    return result

if __name__ == "__main__":
    # 生产环境应该使用配置，这里为了直接运行
    uvicorn.run("main:app", host="192.168.1.114", port=8000, reload=True)