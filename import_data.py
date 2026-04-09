# import_data.py —— 适配你完整的 data.csv（2026年3月19日）
from neo4j import GraphDatabase
import csv
import os

# === 配置 ===
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
CSV_FILE = "data.csv"

def main():
    # 检查 CSV 是否存在
    if not os.path.exists(CSV_FILE):
        print(f"❌ 错误：找不到 {CSV_FILE} 文件！")
        return

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), encrypted=False)
    
    with driver.session() as session:
        # 🔥 第一步：彻底清空旧数据（关键！）
        session.run("MATCH (g:GemCut) DETACH DELETE g")
        print("🗑️  已清空所有 GemCut 节点")

        # 🔥 第二步：读取 CSV 并导入
        with open(CSV_FILE, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                name = row.get("name", "").strip()
                if not name:
                    continue  # 跳过空行
                
                # ✅ 核心逻辑：将 image_files 字符串转为 Python 列表
                image_files_str = row.get("image_files", "")
                image_files = []
                if image_files_str:
                    image_files = [
                        img.strip()
                        for img in image_files_str.split(",")
                        if img.strip()
                    ]
                
                # 构建属性字典（全部字段）
                props = {
                    "name": name,
                    "english_name": row.get("english_name", "").strip(),
                    "image_files": image_files,  # ← 直接传入 list，不是字符串！
                    "structure_description": row.get("structure_description", "").strip(),
                    "advantages": row.get("advantages", "").strip(),
                    "disadvantages": row.get("disadvantages", "").strip(),
                    "suitable_materials": row.get("suitable_materials", "").strip(),
                    "history": row.get("history", "").strip()
                }
                
                # 创建节点
                session.run("""
                    CREATE (g:GemCut $props)
                """, props=props)
                
                count += 1
                print(f"✅ 已导入: {name} | 图片数量: {len(image_files)}")
        
        print(f"\n🎉 共成功导入 {count} 条 GemCut 数据！")

if __name__ == "__main__":
    main()
