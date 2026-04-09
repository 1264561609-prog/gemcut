import os
import json
import re
import time
import streamlit as st
from neo4j import GraphDatabase
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from dotenv import load_dotenv

# ================= 配置区域 =================
load_dotenv()
# Neo4j 配置
NEO4J_CONFIG = {
    "uri": os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687"),
    "username": os.getenv("NEO4J_USERNAME", "neo4j"),
    "password": os.getenv("NEO4J_PASSWORD"),
    "database": "neo4j"
}

# DeepSeek API 配置
DEEPSEEK_CONFIG = {
    "api_key": os.getenv("DEEPSEEK_API_KEY"),
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com/v1"
}

# 琢型图纸库配置
BLUEPRINTS_CONFIG = {
    "root_dir": os.path.join(os.path.dirname(__file__), "gem_blueprints")
}

# 创建目录
os.makedirs(BLUEPRINTS_CONFIG["root_dir"], exist_ok=True)

# 应用配置
APP_CONFIG = {
    "document_dir": "./documents",
    "chunk_size": 800,
    "chunk_overlap": 100
}

# 创建文档目录
os.makedirs(APP_CONFIG["document_dir"], exist_ok=True)

# AI识图配置 - 修复路径问题
AI_VISION_CONFIG = {
    "FEATURES_FILE": "features.pkl",
    "model_name": "dinov2_vits14",
    "IMAGE_BASE_DIR": os.path.join(os.path.dirname(__file__), "images")  # 使用相对于app.py的路径
    # 或者使用绝对路径:
    # "IMAGE_BASE_DIR": r"D:\zhuomian\2207051004\data\image_service\images"
}

# ================= Neo4j 连接函数 =================
def get_neo4j_driver():
    """获取 Neo4j 数据库驱动连接（支持中文）"""
    try:
        from neo4j import GraphDatabase, exceptions
        
        uri = NEO4J_CONFIG["uri"]
        username = NEO4J_CONFIG["username"]
        password = NEO4J_CONFIG["password"]
        
        st.info(f"🔍 尝试连接 Neo4j (支持中文): {uri}")
        
        # 🌟 配置Neo4j支持中文
        driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
            max_connection_lifetime=30 * 60,
            max_connection_pool_size=50,
            connection_timeout=30,
            # 🌟 添加中文编码支持
            connection_acquisition_timeout=60,
        )
        
        # 验证连接
        with driver.session(database=NEO4J_CONFIG["database"]) as session:
            # 🌟 测试中文插入
            test_result = session.run("""
            MERGE (test:TestEntity {name: '中文测试'})
            SET test.test_property = '中文属性值'
            RETURN test.name as name
            """)
            
            test_name = test_result.single()["name"]
            st.success(f"✅ Neo4j 连接成功！中文测试: '{test_name}'")
            
            # 🌟 删除测试节点
            session.run("MATCH (test:TestEntity {name: '中文测试'}) DETACH DELETE test")
            
            return driver
            
    except Exception as e:
        st.error(f"❌ Neo4j 连接失败 (中文支持): {e}")
        st.warning("""
        🔧 **中文支持解决方法**:
        1. 确保Neo4j服务器配置支持UTF-8编码
        2. 在neo4j.conf中添加: dbms.connector.bolt.address=0.0.0.0:7687
        3. 确保Neo4j版本 >= 4.0 (对中文支持更好)
        4. 检查防火墙设置是否允许连接
        """)
        return None

# ================= DeepSeek API 客户端 =================
def init_deepseek_client():
    """初始化 DeepSeek API 客户端"""
    try:
        from openai import OpenAI
        
        if not DEEPSEEK_CONFIG["api_key"]:
            st.warning("⚠️ 未设置 DeepSeek API Key")
            return None
        
        client = OpenAI(
            api_key=DEEPSEEK_CONFIG["api_key"],
            base_url=DEEPSEEK_CONFIG["base_url"]
        )
        
        # 测试连接
        response = client.chat.completions.create(
            model=DEEPSEEK_CONFIG["model"],
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=10
        )
        
        st.success("✅ DeepSeek API 连接成功！")
        return client
        
    except ImportError:
        st.error("❌ 未安装 openai 包，请运行: pip install openai")
        st.code("pip install openai", language="bash")
        return None
    except Exception as e:
        st.error(f"❌ DeepSeek API 连接失败: {e}")
        st.warning("""
        🔧 **解决方法**:
        1. 检查API Key是否正确
        2. 确保网络连接正常
        3. 可以从 https://platform.deepseek.com/ 获取API Key
        """)
        return None

@st.cache_data
def load_feature_db():
    """加载特征库"""
    # ✅ 使用绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    features_path = os.path.join(current_dir, "features.pkl")
    
    print(f"🔍 在 app.py 中查找特征库: {features_path}")
    
    if not os.path.exists(features_path):
        print(f"❌ 在 app.py 中找不到特征库: {features_path}")
        # 列出当前目录内容
        print(f"\n📁 当前目录内容 ({os.getcwd()}):")
        for item in os.listdir(os.getcwd()):
            print(f"   - {item}")
        return None
    
    try:
        with open(features_path, "rb") as f:
            features = pickle.load(f)
        print(f"✅ 在 app.py 中成功加载特征库，包含 {len(features)} 个样本")
        return features
    except Exception as e:
        print(f"❌ 在 app.py 中加载特征库失败: {e}")
        return None

# ---------------------------------------------------------
# 1. 模型加载函数 (使用 Facebook 官方 DINOv2)
# ---------------------------------------------------------
@st.cache_resource
def load_dinov2_model():
    """
    加载 Facebook 原版 DINOv2 模型 (dinov2_vits14)。
    使用 torch.hub 确保与特征提取脚本一致。
    """
    print("👁️ 正在加载 DINOv2 (dinov2_vits14)...")
    
    try:
        # 设置镜像源
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        # 🔥 修复：使用正确的预处理流程
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # 使用 torch.hub 加载官方原版模型
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', 
                             pretrained=True, trust_repo=True)
        
        model.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        print(f"✅ 模型加载成功！设备：{device.upper()}")
        return model, transform, device
        
    except Exception as e:
        st.error(f"❌ 模型加载失败：{e}")
        st.info("💡 请检查网络连接，或尝试离线模式")
        return None, None, None
# ---------------------------------------------------------
# 2. 特征提取函数 (适配 DINOv2 输出结构)
# ---------------------------------------------------------
def get_image_embedding(image_input, model, transform, device):
    """
    读取图片，预处理，并通过模型提取特征向量。
    :param image_input: 可以是文件路径 (str) 或 PIL.Image 对象
    """
    import torch
    
    try:
        # 1. 读取并转换图片
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        else:
            image = image_input.convert('RGB')
        
        # 2. 预处理
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # 3. 推理 (不计算梯度)
        with torch.no_grad():
            # DINOv2 的输出是一个字典，包含 'x_norm_clstoken'
            output_dict = model.forward_features(input_tensor)
            
            # 提取 CLS token 作为整图特征
            embedding = output_dict['x_norm_clstoken']
            
            # 转为 CPU numpy 格式以便后续 sklearn 计算
            return embedding.cpu().squeeze().numpy()
            
    except Exception as e:
        st.error(f"提取特征失败：{e}")
        return None


def display_recognition_results(result):
    """显示识别结果"""
    st.subheader("识别结果")
    
    # 创建结果表格
    results_data = []
    for i, (match, score) in enumerate(zip(result['top_matches'], result['confidence_scores'])):
        confidence = f"{score:.1%}"
        results_data.append({
            "排名": i+1,
            "切工类型": match.replace('_', ' ').title(),
            "置信度": confidence,
            "匹配度": "⭐⭐⭐⭐⭐" if score > 0.8 else "⭐⭐⭐⭐" if score > 0.6 else "⭐⭐⭐"
        })
    
    st.table(results_data)
    
    # 显示最佳匹配
    best_match = result['top_matches'][0]
    best_score = result['confidence_scores'][0]
    st.success(f"最佳匹配: **{best_match.replace('_', ' ').title()}** (置信度: {best_score:.1%})")
    
    # 显示匹配图片（如果有）
    gem_images = {
        "round_brilliant": "round_brilliant.jpg",
        "oval_brilliant": "oval_brilliant.jpg",
        "pear_shape": "pear_shape.jpg",
        "heart_shape": "heart_shape.jpg",
        "emerald_cut": "emerald_cut.jpg"
    }
    
    if best_match in gem_images:
        image_path = os.path.join("gem_blueprints", gem_images[best_match])
        if os.path.exists(image_path):
            st.image(image_path, caption=f"{best_match.replace('_', ' ').title()} 切工示意图", width=400)
    
# ================= 琢型图纸库相关函数 =================
def get_refractive_index_folders():
    """获取所有折射率范围文件夹"""
    ri_folders = []
    blueprints_root = BLUEPRINTS_CONFIG["root_dir"]
    
    if not os.path.exists(blueprints_root):
        st.error(f"❌ 错误：未找到图纸根目录 `{blueprints_root}`。请确保它与 `app.py` 在同一级目录下。")
        return []
    
    for item in os.listdir(blueprints_root):
        full_path = os.path.join(blueprints_root, item)
        if os.path.isdir(full_path) and item.startswith("RI="):
            # 额外验证格式：RI=1.50-1.60 这样的结构
            try:
                # 尝试解析：去掉 "RI="，再按 "-" 分割
                parts = item[3:].split('-')
                if len(parts) == 2 and float(parts[0]) and float(parts[1]):
                    ri_folders.append(item)
            except (ValueError, IndexError):
                continue  # 不符合格式的跳过
    
    return sorted(ri_folders, key=lambda x: float(x[3:].split('-')[0]))  # 按起始RI值排序

def get_cut_type_folders(selected_ri):
    """获取指定折射率下的所有琢型文件夹"""
    ri_path = os.path.join(BLUEPRINTS_CONFIG["root_dir"], selected_ri)
    cut_type_folders = []
    
    if os.path.exists(ri_path):
        for item in os.listdir(ri_path):
            full_path = os.path.join(ri_path, item)
            if os.path.isdir(full_path):
                cut_type_folders.append(item)
    
    return sorted(cut_type_folders)  # 按拼音/字典序排序

def get_blueprint_images(selected_ri, selected_cut):
    """获取指定琢型下的所有图纸图片"""
    cut_path = os.path.join(BLUEPRINTS_CONFIG["root_dir"], selected_ri, selected_cut)
    image_files = []
    
    if os.path.exists(cut_path):
        for item in os.listdir(cut_path):
            if item.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_files.append(item)
    
    return sorted(image_files)

# ================= 三元组提取核心函数 =================
def extract_triples_with_llm(text: str, client, max_retries: int = 3) -> list:
    """
    使用LLM从文本中提取三元组（中文优化版本）
    """
    try:
        if not text or len(text.strip()) < 50:
            st.warning("⚠️ 文本内容过短，无法提取有效三元组")
            return []
        
        # 🌟 中文优化的提示词
        extraction_prompt = f"""
        你是一位宝石切工领域的中文专家，需要从以下中文文本中精确提取知识三元组。
        
        【文本内容】
        {text}
        
        【提取要求】
        1. 仔细阅读文本，识别所有有意义的中文实体和它们之间的关系
        2. 每个三元组格式必须为：(主体, 关系, 客体)
        3. 只提取文本中明确提到或强烈暗示的关系
        4. 确保主体和客体是具体的中文实体（人物姓名、宝石类型、切工名称、时间、地点等）
        5. 关系必须是中文动词或动词短语（例如："发明"、"演变自"、"具有属性"、"适用于"、"发现于"、"流行于"等）
        6. 为每个三元组评估置信度（0.0-1.0）
        7. 保留原文中的中文术语，不要翻译成英文
        
        【重要示例】
        ✅ 正确示例:
        (玫瑰花琢型, 发明, 16世纪)
        (明亮式切工, 适用于, 钻石)
        (马歇尔·托尔可夫斯基, 改进, 理想式切工)
        (祖母绿切工, 具有属性, 阶梯状刻面)
        
        ❌ 错误示例:
        (Rose Cut, invented, 16th century)  // 不要使用英文
        (Brilliant Cut, suitable for, Diamond)  // 不要使用英文
        
        【返回格式】
        请严格按照以下JSON格式返回，只使用中文，不要包含任何其他文本：
        {{
            "triples": [
                {{
                    "subject": "主体实体（中文）",
                    "predicate": "关系（中文动词）",
                    "object": "客体实体（中文）",
                    "confidence": 0.95,
                    "evidence": "支持该三元组的原文片段（中文）"
                }}
            ]
        }}
        
        【重要提醒】
        - 只返回JSON格式，不要包含```json等标记
        - 确保使用中文字符，不要混入英文术语
        - 如果没有找到有效三元组，返回空数组
        - 确保JSON语法正确，特别是中文引号要使用英文双引号
        """
        
        if not client:
            st.error("❌ DeepSeek 客户端未初始化")
            return []
        
        # 带重试机制的API调用
        for attempt in range(max_retries):
            try:
                st.info(f"🔄 三元组提取尝试 {attempt + 1}/{max_retries} - 处理中文文本")
                
                response = client.chat.completions.create(
                    model=DEEPSEEK_CONFIG["model"],
                    messages=[
                        {"role": "system", "content": "你是一位宝石切工领域的中文专家，严格按JSON格式返回中文结果，只使用中文，不要包含任何其他文本。"},
                        {"role": "user", "content": extraction_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1500,
                    timeout=60
                )
                
                response_text = response.choices[0].message.content.strip()
                
                # 🌟 中文调试信息
                with st.expander("🔍 调试信息 - 原始API响应（中文）", expanded=False):
                    st.code(response_text, language="text")
                    st.write(f"响应长度: {len(response_text)} 字符")
                
                # 🌟 增强的中文JSON提取
                cleaned_response = extract_json_from_text(response_text)
                
                if not cleaned_response:
                    # 🌟 尝试直接修复中文JSON
                    cleaned_response = fix_chinese_json(response_text)
                
                if cleaned_response:
                    try:
                        result = json.loads(cleaned_response)
                        triples = result.get('triples', [])
                        
                        if triples:
                            # 🌟 验证和过滤中文三元组
                            valid_triples = []
                            for triple in triples:
                                subject = str(triple.get('subject', '')).strip()
                                predicate = str(triple.get('predicate', '')).strip()
                                object_ = str(triple.get('object', '')).strip()
                                confidence = float(triple.get('confidence', 0.0))
                                
                                # 🌟 中文验证规则
                                if (subject and predicate and object_ and 
                                    confidence > 0.5):
                                    
                                    # 🌟 强制转换为中文
                                    valid_triples.append({
                                        "subject": subject,
                                        "predicate": predicate,
                                        "object": object_,
                                        "confidence": confidence,
                                        "evidence": str(triple.get('evidence', ''))
                                    })
                            
                            if valid_triples:
                                st.success(f"✅ 成功提取 {len(valid_triples)} 个中文三元组")
                                # 🌟 显示提取的三元组
                                with st.expander("📋 提取的中文三元组", expanded=False):
                                    for i, triple in enumerate(valid_triples):
                                        st.write(f"{i+1}. ({triple['subject']}, {triple['predicate']}, {triple['object']}) - 置信度: {triple['confidence']:.2f}")
                                return valid_triples
                    except json.JSONDecodeError as e:
                        st.warning(f"⚠️ JSON解析失败: {e}")
                        st.warning(f"问题JSON: {cleaned_response[:200]}")
                
                # 重试前等待
                if attempt < max_retries - 1:
                    time.sleep(2)
            
            except Exception as api_error:
                st.warning(f"⚠️ API调用失败 ({attempt + 1}/{max_retries}): {api_error}")
                if attempt < max_retries - 1:
                    time.sleep(3)
        
        st.error("❌ 中文三元组提取彻底失败，达到最大重试次数")
        return []
    
    except Exception as e:
        st.error(f"❌ 中文三元组提取过程发生意外错误: {e}")
        return []

# 🌟 新增JSON修复函数
def fix_chinese_json(text):
    """
    修复包含中文的JSON字符串
    """
    try:
        # 方法1：尝试修复常见的中文JSON问题
        fixed_text = text
        
        # 修复中文引号问题
        fixed_text = fixed_text.replace('“', '"').replace('”', '"')
        fixed_text = fixed_text.replace('‘', "'").replace('’', "'")
        
        # 修复中文冒号/逗号问题
        fixed_text = re.sub(r'：', ':', fixed_text)
        fixed_text = re.sub(r'，', ',', fixed_text)
        
        # 提取JSON部分
        json_match = re.search(r'\{[\s\S]*\}', fixed_text)
        if json_match:
            potential_json = json_match.group(0)
            try:
                json.loads(potential_json)
                return potential_json
            except json.JSONDecodeError:
                pass
        
        # 方法2：手动构建JSON
        if '"triples"' in fixed_text or '“triples”' in fixed_text:
            st.warning("⚠️ 尝试手动修复中文JSON")
            # 简单的三元组提取
            triples = []
            
            # 寻找 (主体, 关系, 客体) 模式
            pattern = r'   $  ([^,]+),\s*([^,]+),\s*([^)]+)  $   '
            matches = re.findall(pattern, fixed_text)
            
            for match in matches:
                subject, predicate, object_ = match
                subject = subject.strip().strip('"').strip("'")
                predicate = predicate.strip().strip('"').strip("'")
                object_ = object_.strip().strip('"').strip("'")
                
                triples.append({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object_,
                    "confidence": 0.8,
                    "evidence": "手动提取"
                })
            
            if triples:
                result = {"triples": triples}
                return json.dumps(result, ensure_ascii=False)
        
        return None
    
    except Exception as e:
        st.error(f"❌ JSON修复失败: {e}")
        return None

def create_new_document_graph(session, triples, source_doc):
    """首次处理文档：创建新知识图谱"""
    try:
        # 1. 创建文档节点
        session.run("""
        MERGE (doc:Document {source_name: $source_name})
        SET doc.processed_at = datetime(),
            doc.triples_count = $triples_count,
            doc.status = 'active'
        """, parameters={
            'source_name': source_doc,
            'triples_count': len(triples)
        })
        
        # 2. 批量处理三元组
        session.run("""
        UNWIND $triples AS triple
        // 创建或合并主体实体
        MERGE (subject:Entity {name: triple.subject})
        ON CREATE SET 
            subject.type = '未知',
            subject.confidence = triple.confidence,
            subject.created_at = datetime(),
            subject.sources = [$source_doc]
        ON MATCH SET
            subject.confidence = (subject.confidence + triple.confidence) / 2,
            subject.sources = CASE 
                WHEN NOT $source_doc IN subject.sources 
                THEN subject.sources + [$source_doc] 
                ELSE subject.sources 
            END
        
        // 创建或合并客体实体
        MERGE (object:Entity {name: triple.object})
        ON CREATE SET 
            object.type = '未知',
            object.confidence = triple.confidence,
            object.created_at = datetime(),
            object.sources = [$source_doc]
        ON MATCH SET
            object.confidence = (object.confidence + triple.confidence) / 2,
            object.sources = CASE 
                WHEN NOT $source_doc IN object.sources 
                THEN object.sources + [$source_doc] 
                ELSE object.sources 
            END
        
        // 创建关系
        MERGE (subject)-[r:RELATIONSHIP {type: triple.predicate, source: $source_doc}]->(object)
        ON CREATE SET 
            r.confidence = triple.confidence,
            r.evidence = triple.evidence,
            r.created_at = datetime(),
            r.sources = [$source_doc]
        ON MATCH SET
            r.confidence = (r.confidence + triple.confidence) / 2,
            r.evidence = COALESCE(r.evidence, '') + ' | ' + triple.evidence,
            r.sources = CASE 
                WHEN NOT $source_doc IN r.sources 
                THEN r.sources + [$source_doc] 
                ELSE r.sources 
            END
        """, parameters={
            'triples': triples,
            'source_doc': source_doc
        })
        
        st.success("✅ 首次知识图谱构建完成！")
        return True
        
    except Exception as e:
        st.error(f"❌ 首次构建失败: {e}")
        return False

def update_existing_document(session, triples, source_doc):
    """覆盖更新策略：删除旧数据，重新构建"""
    try:
        # 1. 删除该文档相关的所有数据
        session.run("""
        MATCH (doc:Document {source_name: $source_name})-[r:CONTAINS]->(e:Entity)
        DETACH DELETE doc, r, e
        """, parameters={'source_name': source_doc})
        
        # 2. 重新创建
        st.info("🔄 删除旧数据，重新构建知识图谱...")
        return create_new_document_graph(session, triples, source_doc)
        
    except Exception as e:
        st.error(f"❌ 覆盖更新失败: {e}")
        return False

def append_new_triples(session, triples, source_doc):
    """增量添加策略：只添加新关系"""
    try:
        # 1. 获取已存在的实体和关系
        existing_entities = session.run("""
        MATCH (e:Entity)
        WHERE $source_doc IN e.sources
        RETURN e.name as name
        """, parameters={'source_doc': source_doc})
        
        existing_entity_names = [record["name"] for record in existing_entities]
        
        # 2. 只处理不重复的三元组
        new_triples = []
        for triple in triples:
            subject = triple["subject"]
            object_ = triple["object"]
            
            if subject not in existing_entity_names or object_ not in existing_entity_names:
                new_triples.append(triple)
        
        if not new_triples:
            st.warning("⚠️ 没有发现新的三元组需要添加")
            return True
        
        # 3. 添加新三元组
        session.run("""
        UNWIND $triples AS triple
        MERGE (subject:Entity {name: triple.subject})
        ON CREATE SET 
            subject.type = '未知',
            subject.confidence = triple.confidence,
            subject.created_at = datetime()
        SET subject.sources = COALESCE(subject.sources, []) + [$source_doc]
        
        MERGE (object:Entity {name: triple.object})
        ON CREATE SET 
            object.type = '未知',
            object.confidence = triple.confidence,
            object.created_at = datetime()
        SET object.sources = COALESCE(object.sources, []) + [$source_doc]
        
        MERGE (subject)-[r:RELATIONSHIP {type: triple.predicate, source: $source_doc}]->(object)
        ON CREATE SET 
            r.confidence = triple.confidence,
            r.evidence = triple.evidence,
            r.created_at = datetime()
        SET r.sources = COALESCE(r.sources, []) + [$source_doc]
        """, parameters={
            'triples': new_triples,
            'source_doc': source_doc
        })
        
        st.success(f"✅ 增量添加完成！新增 {len(new_triples)} 个三元组")
        return True
        
    except Exception as e:
        st.error(f"❌ 增量添加失败: {e}")
        return False

# ================= 知识图谱构建函数 =================
def build_knowledge_graph_from_triples(driver, triples: list, source_doc: str = "unknown"):
    """
    重构后的知识图谱构建函数 - 支持重复文档处理
    """
    try:
        if not triples:
            st.warning("⚠️ 没有三元组需要构建到知识图谱")
            return False
        
        with driver.session(database=NEO4J_CONFIG["database"]) as session:
            st.info("🎯 准备构建知识图谱...")
            
            # 🔄 新策略：先检查文档是否已处理
            doc_exists = session.run("""
            MATCH (doc:Document {source_name: $source_name}) 
            RETURN count(doc) as exists
            """, parameters={'source_name': source_doc}).single()["exists"]
            
            if doc_exists > 0:
                st.warning(f"⚠️ 文档 '{source_doc}' 已处理过。选择处理方式：")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("🔄 覆盖更新", key=f"overwrite_{source_doc}"):
                        return update_existing_document(session, triples, source_doc)
                with col2:
                    if st.button("➕ 增量添加", key=f"append_{source_doc}"):
                        return append_new_triples(session, triples, source_doc)
                with col3:
                    if st.button("❌ 取消", key=f"cancel_{source_doc}"):
                        return False
                
                st.info("💡 提示：覆盖更新会替换旧数据，增量添加只添加新关系")
                return False
            
            # 🆕 首次处理：正常构建
            success = create_new_document_graph(session, triples, source_doc)
            return success
            
    except Exception as e:
        st.error(f"❌ 知识图谱构建失败: {e}")
        st.error(f"错误详情: {type(e).__name__}")
        st.code(str(e), language='python')
        return False

# ================= 智能问答函数 =================
def query_knowledge_graph(question: str, driver, client):
    """
    基于知识图谱的智能问答（中文优化版本）
    """
    try:
        if not question.strip():
            return "请提供有效的问题。"
                
        st.info(f"🔍 处理中文问题: '{question}'")
        
        # 🌟 中文实体识别提示词
        entity_prompt = f"""
        你是一位宝石切工领域的中文专家，需要从以下中文问题中精确识别关键实体。
        
        【用户问题】
        {question}
        
        【识别要求】
        1. 识别问题中提到的所有关键中文实体（如：切工名称、人物姓名、宝石类型、技术术语等）
        2. 只返回与宝石切工领域相关的中文实体
        3. 如果问题中没有明确提到实体，尝试推断最相关的中文实体
        4. 为每个实体评估相关性分数（0.0-1.0）
        5. 实体名称必须使用原文中的中文术语
        
        【返回格式】
        请严格按照以下JSON格式返回，只使用中文，不要包含任何其他文本：
        {{
            "entities": [
                {{
                    "name": "实体名称（中文）",
                    "relevance": 0.95,
                    "type": "实体类型（如：切工、人物、宝石等）"
                }}
            ],
            "main_entity": "最重要的一个实体名称（中文）",
            "query_type": "查询类型（如：定义、历史、特点、关系等）"
        }}
        
        【重要提醒】
        - 只返回JSON格式，不要包含```json等标记
        - 确保使用中文字符，不要混入英文术语
        - 确保JSON语法正确，特别是中文引号要使用英文双引号
        """
        
        # 带重试机制的实体识别
        entity_data = None
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                st.info(f"🔄 实体识别尝试 {attempt + 1}/{max_retries}")
                
                entity_response = client.chat.completions.create(
                    model=DEEPSEEK_CONFIG["model"],
                    messages=[
                        {"role": "system", "content": "你是一个宝石切工领域的实体识别专家，严格按JSON格式返回结果。"},
                        {"role": "user", "content": entity_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500,
                    timeout=30
                )
                
                response_text = entity_response.choices[0].message.content.strip()
                
                # 调试信息 - 显示原始响应
                with st.expander("🔍 调试信息 - 原始API响应", expanded=False):
                    st.code(response_text[:1000], language="text")
                
                # 智能清理和提取JSON
                cleaned_response = extract_json_from_text(response_text)
                
                if cleaned_response:
                    entity_data = json.loads(cleaned_response)
                    break
                else:
                    st.warning(f"⚠️ 第{attempt + 1}次尝试：未找到有效JSON，重试中...")
            
            except json.JSONDecodeError as e:
                st.warning(f"⚠️ JSON解析失败 (尝试 {attempt + 1}): {e}")
                st.warning(f"无效响应: '{response_text[:200]}'")
            except Exception as api_error:
                st.warning(f"⚠️ API调用失败 (尝试 {attempt + 1}): {api_error}")
            
            if attempt < max_retries - 1:
                time.sleep(2)
        
        if not entity_data:
            st.error("❌ 实体识别彻底失败，使用备用策略")
            # 备用策略：使用简单文本分析
            entities = extract_entities_fallback(question)
            if entities:
                entity_data = {
                    "entities": [{"name": ent, "relevance": 0.8, "type": "未知"} for ent in entities],
                    "main_entity": entities[0],
                    "query_type": "general"
                }
            else:
                return "抱歉，无法理解问题中的关键实体。请尝试重新表述问题，例如：'玫瑰花琢型的特点是什么？' 或 '谁发明了明亮式切工？'"
        
        entities = entity_data.get('entities', [])
        main_entity = entity_data.get('main_entity', "")
        query_type = entity_data.get('query_type', "general")
        
        if not entities and not main_entity:
            st.warning("⚠️ 未识别到有效实体，使用问题关键词")
            # 从问题中提取关键词作为实体
            keywords = extract_keywords_from_question(question)
            if keywords:
                main_entity = keywords[0]
                entities = [{"name": k, "relevance": 0.7, "type": "关键词"} for k in keywords]
            else:
                return "抱歉，无法从问题中提取关键信息。请尝试更具体的问题，例如：'玫瑰花琢型是谁发明的？'"
        
        st.success(f"✅ 识别到关键实体: {main_entity}")
        if entities:
            entity_names = [ent.get('name', '') for ent in entities if ent.get('name')]
            st.info(f"🔍 相关实体: {', '.join(entity_names)}")
        
        # 2. 从知识图谱中检索相关信息
        graph_context = ""
        retrieved_relations = []
        
        with driver.session(database=NEO4J_CONFIG["database"]) as session:
            # 检索实体及其关系
            entity_names = [main_entity] + [ent.get('name', '') for ent in entities]
            entity_names = list(set([name.strip() for name in entity_names if name.strip()]))
            
            if not entity_names:
                return "抱歉，未识别到有效的查询实体。"
            
            st.info(f"📡 从知识图谱检索实体: {', '.join(entity_names)}")
            
            result = session.run("""
            MATCH (e:Entity)-[r:RELATIONSHIP]-(related:Entity)
            WHERE e.name IN $entity_names OR related.name IN $entity_names
            RETURN 
                e.name as subject,
                r.type as predicate, 
                related.name as object,
                r.confidence as confidence,
                r.evidence as evidence,
                r.sources as sources
            ORDER BY r.confidence DESC
            LIMIT 15
            """, parameters={'entity_names': entity_names})
            
            retrieved_data = list(result)
            
            if retrieved_data:
                retrieved_relations = retrieved_data
                context_parts = []
                for i, record in enumerate(retrieved_data):
                    context_parts.append(
                        f"关系 {i+1}:\n"
                        f"主体: {record['subject']}\n"
                        f"关系: {record['predicate']}\n"
                        f"客体: {record['object']}\n"
                        f"置信度: {record['confidence']:.2f}\n"
                        f"证据: {record['evidence']}\n"
                        f"来源: {', '.join(record['sources']) if record['sources'] else '未知'}\n"
                    )
                graph_context = "\n".join(context_parts)
                st.success(f"✅ 从知识图谱找到 {len(retrieved_data)} 个相关关系")
            else:
                st.warning(f"⚠️ 未在知识图谱中找到与 '{main_entity}' 直接相关的关系")
                # 尝试模糊匹配
                fuzzy_result = session.run("""
                MATCH (e:Entity)-[r:RELATIONSHIP]-(related:Entity)
                WHERE toLower(e.name) CONTAINS toLower($keyword) OR toLower(related.name) CONTAINS toLower($keyword)
                RETURN 
                    e.name as subject,
                    r.type as predicate, 
                    related.name as object,
                    r.confidence as confidence,
                    r.evidence as evidence
                ORDER BY r.confidence DESC
                LIMIT 10
                """, parameters={'keyword': main_entity[:10]})
                
                fuzzy_data = list(fuzzy_result)
                if fuzzy_data:
                    retrieved_relations = fuzzy_data
                    st.info(f"🔍 通过模糊匹配找到 {len(fuzzy_data)} 个相关关系")
                    context_parts = []
                    for i, record in enumerate(fuzzy_data):
                        context_parts.append(
                            f"关系 {i+1} (模糊匹配):\n"
                            f"主体: {record['subject']}\n"
                            f"关系: {record['predicate']}\n"
                            f"客体: {record['object']}\n"
                            f"置信度: {record['confidence']:.2f}\n"
                            f"证据: {record['evidence']}\n"
                        )
                    graph_context = "\n".join(context_parts)
                else:
                    st.warning("⚠️ 知识图谱中未找到相关信息，将使用LLM的基础知识回答")
                    graph_context = "未检索到相关信息，基于LLM的专业知识回答。"
        
        # 🌟 中文回答生成
        answer_prompt = f"""
        你是一位宝石切工领域的中文权威专家，基于以下中文专业知识回答用户的中文问题。
        
        【检索到的中文专业知识】
        {graph_context}
        
        【用户原始问题】
        {question}
        
        【回答要求】
        1. **使用中文回答**：整个回答必须使用中文，不要混入英文术语
        2. **专业术语**：使用宝石切工领域的标准中文术语
        3. **结构清晰**：
           - 先给出直接的中文答案
           - 然后提供详细的中文解释
           - 最后可以补充相关背景或应用
        4. **准确性**：基于检索到的专业知识回答，不要编造信息
        5. **权威性**：使用专业的语气，体现专家水平
        
        【禁止行为】
        - 不要使用英文术语（如Brilliant Cut，应该用"明亮式切工"）
        - 不要提及你是一个AI或语言模型
        - 不要提供与宝石切工无关的信息
        
        请用中文专业、权威、清晰地回答：
        """
        
        # 带重试的回答生成
        answer = ""
        for attempt in range(2):  # 减少重试次数
            try:
                st.info(f"🤖 生成专业回答 (尝试 {attempt + 1}/2)")
                
                answer_response = client.chat.completions.create(
                    model=DEEPSEEK_CONFIG["model"],
                    messages=[
                        {"role": "system", "content": "你是一个宝石切工领域的权威专家，提供准确、专业的知识解答。"},
                        {"role": "user", "content": answer_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1024,
                    timeout=60
                )
                
                answer = answer_response.choices[0].message.content.strip()
                
                if answer:
                    break
                else:
                    st.warning("⚠️ 生成空回答，重试中...")
            except Exception as gen_error:
                st.warning(f"⚠️ 回答生成失败 (尝试 {attempt + 1}): {gen_error}")
        
        if not answer:
            # 备用回答
            answer = generate_fallback_answer(question, main_entity, query_type)
        
        # 4. 添加来源信息和置信度
        if retrieved_relations:
            confidence_score = sum(float(record['confidence']) for record in retrieved_relations) / len(retrieved_relations)
            confidence_text = get_confidence_text(confidence_score)
            
            source_info = "\n\n---\n🔍 **回答置信度**: " + confidence_text + "\n"
            source_info += f"📊 **知识来源**: 基于知识图谱中 {len(retrieved_relations)} 个相关关系\n"
            source_info += f"💎 **核心实体**: {main_entity}\n"
            source_info += f"🕐 **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            answer += source_info
        else:
            answer += "\n\n---\n⚠️ **注意**: 未在知识图谱中找到直接相关信息，此回答基于LLM的专业知识生成。建议上传相关文档以丰富知识库。"
        
        st.success("✅ 专业回答生成完成")
        return answer
    
    except Exception as e:
        st.error(f"❌ 问答过程中发生错误: {e}")
        st.error(f"错误类型: {type(e).__name__}")
        st.code(str(e), language='python')
        
        # 提供友好的错误回复
        return f"抱歉，在回答问题时遇到了技术问题。请稍后重试，或尝试重新表述您的问题。错误详情已记录。"

# ================= 新增的辅助函数 =================
def extract_json_from_text(text):
    """
    从文本中智能提取JSON内容
    """
    try:
        # 方法1：直接解析（如果整个文本是JSON）
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass
        
        # 方法2：查找JSON代码块
        json_pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(json_pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            cleaned_json = matches[0].strip()
            try:
                json.loads(cleaned_json)
                return cleaned_json
            except json.JSONDecodeError:
                pass
        
        # 方法3：查找大括号内的内容
        brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(brace_pattern, text, re.DOTALL)
        for match in matches:
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue
        
        # 方法4：清理并尝试解析
        cleaned = re.sub(r'^[^{]*', '', text)  # 移除开头的非{字符
        cleaned = re.sub(r'[^}]*$', '', cleaned)  # 移除结尾的非}字符
        if cleaned.startswith('{') and cleaned.endswith('}'):
            try:
                json.loads(cleaned)
                return cleaned
            except json.JSONDecodeError:
                pass
        
        st.warning("⚠️ 未找到有效的JSON内容")
        st.warning(f"原始文本: '{text[:200]}'")
        return None
    
    except Exception as e:
        st.error(f"❌ JSON提取过程中出错: {e}")
        return None

def extract_entities_fallback(question):
    """
    备用实体提取方法（当API失败时）
    """
    try:
        import jieba
        # 简单的中文分词
        keywords = jieba.lcut(question)
        # 过滤掉常见停用词
        stop_words = ['什么', '谁', '的', '是', '在', '有', '和', '与', '或', '吗', '呢', '吧', '了', '的']
        entities = [word for word in keywords 
                   if len(word) > 1 and word not in stop_words 
                   and not word.isdigit()]
        
        if entities:
            st.info(f"🔧 使用备用方法提取实体: {', '.join(entities)}")
            return entities[:3]  # 返回最多3个实体
        
        return []
    except ImportError:
        # 如果没有安装jieba，使用简单方法
        st.warning("⚠️ 未安装jieba库，使用简单关键词提取")
        keywords = [word for word in question.split() if len(word) > 1]
        return keywords[:3]

def extract_keywords_from_question(question):
    """
    从问题中提取关键词
    """
    # 简单的关键词提取
    important_words = []
    
    # 移除标点符号
    cleaned = re.sub(r'[^\w\s]', '', question)
    
    # 按空格和常见分隔符分割
    words = re.split(r'[\s\-\/\(\)\"\'\[\]]', cleaned)
    
    for word in words:
        if len(word) >= 2 and word not in ['什么', '谁', '哪', '是否', '怎么', '如何', '为什么', '什么']:
            important_words.append(word)
    
    return important_words[:5]  # 返回最多5个关键词

def generate_fallback_answer(question, main_entity, query_type):
    """
    生成备用回答（当API失败时）
    """
    st.warning("⚠️ 使用备用回答生成器")
    
    if not main_entity:
        return "抱歉，无法理解您的问题。请尝试更具体的问题，例如：'玫瑰花琢型的特点是什么？'"
    
    if "特点" in question or "特征" in question:
        return f"关于{main_entity}的特点，建议您上传相关的宝石切工文档，以便我能够提供更准确和专业的信息。"
    elif "谁" in question or "发明" in question:
        return f"关于{main_entity}的发明者或历史，建议您上传相关的历史文档或技术资料，这样我可以为您提供准确的信息。"
    elif "什么" in question or "定义" in question:
        return f"{main_entity}是宝石切工领域的一个重要术语。要获得准确的定义和详细信息，请上传相关的专业文档。"
    else:
        return f"关于{main_entity}的问题，我需要更多相关文档来提供准确和专业的回答。建议您先上传相关的宝石切工资料。"

def get_confidence_text(confidence_score):
    """
    根据置信度分数返回文本描述
    """
    if confidence_score >= 0.85:
        return "⭐⭐⭐⭐⭐ (非常可信)"
    elif confidence_score >= 0.7:
        return "⭐⭐⭐⭐ (可信)"
    elif confidence_score >= 0.5:
        return "⭐⭐⭐ (一般可信)"
    elif confidence_score >= 0.3:
        return "⭐⭐ (可信度较低)"
    else:
        return "⭐ (可信度很低)"

# ================= 文档处理函数 =================
def process_document_for_knowledge_graph(file_path: str, driver, client, source_name: str = None):
    """
    改进后的文档处理函数 - 支持重复处理
    """
    try:
        if not source_name:
            source_name = os.path.basename(file_path)
        
        st.info(f"📄 正在处理文档: {source_name}")
        
        # 1. 读取文档（保持不变）
        doc = Document(file_path)
        paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        full_text = "\n".join(paragraphs)
        
        if not full_text.strip():
            st.warning("⚠️ 文档内容为空")
            return False
        
        st.success(f"✅ 成功读取文档，总字符数: {len(full_text)}")
        
        # 2. 文本分割（保持不变）
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=APP_CONFIG["chunk_size"],
            chunk_overlap=APP_CONFIG["chunk_overlap"],
            separators=["\n\n", "\n", "。", "！", "？", "；", "：", " ", ""],
            length_function=len,
        )
        chunks = text_splitter.split_text(full_text)
        st.success(f"✅ 文档分割完成，共 {len(chunks)} 个文本块")
        
        # 3. 提取三元组（保持不变）
        all_triples = []
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:
                continue
            
            triples = extract_triples_with_llm(chunk, client)
            if triples:
                for triple in triples:
                    triple['source_doc'] = source_name
                all_triples.extend(triples)
        
        if not all_triples:
            st.warning("⚠️ 未从文档中提取到有效三元组")
            return False
        
        st.success(f"✅ 共提取 {len(all_triples)} 个三元组")
        
        # 4. 🔄 检查是否已处理过
        if 'processed_docs' not in st.session_state:
            st.session_state.processed_docs = []
        
        # 5. 构建知识图谱（使用新函数）
        success = build_knowledge_graph_from_triples(driver, all_triples, source_name)
        
        if success:
            # 6. 更新会话状态（改进：记录详细信息）
            doc_record = {
                'file_path': file_path,
                'source_name': source_name,
                'processed_at': datetime.now().isoformat(),
                'triples_count': len(all_triples),
                'status': 'success',
                'hash': hash(full_text)  # 添加内容哈希，便于检测重复
            }
            
            # 更新已处理文档列表（去重逻辑）
            updated_docs = []
            for record in st.session_state.processed_docs:
                if record['source_name'] != source_name:
                    updated_docs.append(record)
            updated_docs.append(doc_record)
            st.session_state.processed_docs = updated_docs
            
            st.balloons()
            st.success(f"🎉 文档 '{source_name}' 处理完成！")
            return True
        else:
            st.error("❌ 知识图谱构建失败")
            return False
            
    except Exception as e:
        st.error(f"❌ 文档处理失败: {e}")
        return False
# ================= 工具函数 =================
def save_uploaded_document(uploaded_file):
    """保存上传的文档"""
    try:
        file_path = os.path.join(APP_CONFIG["document_dir"], uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ 文件保存成功: {file_path}")
        return file_path
    except Exception as e:
        st.error(f"❌ 文件保存失败: {e}")
        return None

def get_uploaded_documents():
    """获取已上传的文档列表"""
    try:
        return [f for f in os.listdir(APP_CONFIG["document_dir"]) if f.endswith('.docx')]
    except Exception as e:
        st.error(f"❌ 获取文档列表失败: {e}")
        return []

def get_graph_statistics(driver):
    """获取知识图谱统计信息"""
    if not driver:
        return None
    
    try:
        with driver.session(database=NEO4J_CONFIG["database"]) as session:
            # 获取实体数量
            entity_result = session.run("MATCH (n:Entity) RETURN count(n) as entity_count")
            entity_count = entity_result.single()["entity_count"]
            
            # 获取关系数量
            rel_result = session.run("MATCH ()-[r:RELATIONSHIP]->() RETURN count(r) as rel_count")
            rel_count = rel_result.single()["rel_count"]
            
            # 获取唯一实体类型
            type_result = session.run("""
                MATCH (n:Entity) 
                WHERE n.type IS NOT NULL 
                RETURN DISTINCT n.type as type, count(n) as count
                ORDER BY count DESC
                LIMIT 10
            """)
            types = [{"type": record["type"], "count": record["count"]} for record in type_result]
            
            return {
                "entity_count": entity_count,
                "relationship_count": rel_count,
                "entity_types": types,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
    except Exception as e:
        st.error(f"❌ 获取图谱统计失败: {e}")
        return None

# ================= 页面功能函数 =================
IMAGE_CONFIG = {
    "base_dir": "D:\zhuomian\2207051004\data\image_service\images"
}

def render_home_page():
    """首页"""
    st.title("💎 宝石切工知识图谱系统")
    st.markdown("""
    ## 🌟 系统简介
    
    这是一个基于LLM和知识图谱的宝石切工智能问答系统，具有以下特点：
    
    ✅ **直接三元组提取**：使用大语言模型直接从文档中提取实体关系三元组
    
    ✅ **精准知识构建**：基于三元组构建Neo4j知识图谱，确保知识准确性
    
    ✅ **智能问答**：基于知识图谱进行专业问答，提供权威解答
    
    ✅ **动态更新**：支持实时上传新文档，动态更新知识库
    
    ✅ **AI识图**：基于DINOv2的宝石切工图片识别
    
    ✅ **琢型图纸库**：按折射率分类的宝石琢型图纸检索
    
    ## 🚀 使用指南
    
    1. **📚 文档管理**：上传Word文档，系统自动提取三元组构建知识图谱
    2. **🧠 知识图谱问答**：输入专业问题，获取基于知识图谱的精准回答
    3. **📊 图谱统计**：查看知识图谱的统计信息和结构
    4. **🔍 基础查询**：浏览和搜索宝石切工数据库
    5. **📷 AI识图**：上传宝石图片进行智能识别
    6. **🔍 琢型图纸**：按折射率范围查找宝石琢型图纸
    
    ## 💡 技术架构
    
    - **前端**：Streamlit
    - **LLM**：DeepSeek API
    - **图数据库**：Neo4j
    - **文档处理**：python-docx
    - **计算机视觉**：DINOv2 + PyTorch
    - **特征匹配**：scikit-learn
    
    欢迎使用！开始探索宝石切工的奥秘吧！✨
    """)

def render_document_management_page(driver, client):
    """文档管理页面"""
    st.subheader("📚 文档管理与知识图谱构建")
    st.markdown("""
    📁 **知识构建流程**：
    1. 上传 Word 文档 (.docx)
    2. 使用LLM直接抽取实体关系三元组
    3. 构建Neo4j知识图谱
    4. 支持文档删除和更新
    """)
    
    # API Key 设置
    if not DEEPSEEK_CONFIG["api_key"]:
        api_key = st.text_input("🔑 请输入 DeepSeek API Key", type="password", 
                               help="可以从 https://platform.deepseek.com/ 获取免费 API key")
        if api_key:
            DEEPSEEK_CONFIG["api_key"] = api_key
            st.success("✅ API Key 已设置")
            st.rerun()
    
    # 检查连接状态
    if not driver:
        st.error("❌ 无法连接 Neo4j 数据库。请检查配置。")
        return
    
    if not client:
        st.error("❌ DeepSeek API 未配置。请设置 API Key。")
        return
    
    # 显示当前文档
    uploaded_docs = get_uploaded_documents()
    st.markdown(f"### 📁 已上传文档 ({len(uploaded_docs)} 个)")
    
    if uploaded_docs:
        for doc in uploaded_docs:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"- 📄 `{doc}`")
            with col2:
                if st.button("❌ 删除", key=f"delete_{doc}"):
                    file_path = os.path.join(APP_CONFIG["document_dir"], doc)
                    try:
                        os.remove(file_path)
                        st.success(f"✅ 文档 '{doc}' 已删除！")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 删除失败：{e}")
    
    st.divider()
    
    # 上传新文档
    st.markdown("### 📤 上传新文档")
    uploaded_file = st.file_uploader("上传 Word 文档 (.docx)", type=['docx'])
    
    if uploaded_file:
        st.markdown(f"📄 **文件名**: {uploaded_file.name}")
        st.markdown(f"📊 **文件大小**: {uploaded_file.size / 1024:.1f} KB")
        
        if st.button("🚀 构建知识图谱", type="primary", key="build_kg_btn"):
            with st.spinner("💾 正在保存文档..."):
                file_path = save_uploaded_document(uploaded_file)
            
            if file_path:
                with st.spinner("🔄 正在使用LLM抽取三元组并构建知识图谱..."):
                    success = process_document_for_knowledge_graph(
                        file_path, 
                        driver, 
                        client,
                        uploaded_file.name
                    )
                    
                    if success:
                        st.balloons()
                        st.success("🎉 知识图谱构建完成！现在可以在 '知识图谱问答' 页面进行智能问答。")
                        st.rerun()
                        
    if st.session_state.processed_docs:
        st.markdown("### 📊 已处理文档状态")
        for doc in st.session_state.processed_docs:
            with st.expander(f"📄 {doc['source_name']} ({doc['triples_count']} 个三元组)"):
                st.write(f"**处理时间**: {doc['processed_at']}")
                st.write(f"**状态**: {doc['status']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 重新处理", key=f"reprocess_{doc['source_name']}"):
                        # 重新处理逻辑
                        file_path = os.path.join(APP_CONFIG["document_dir"], doc['source_name'])
                        if os.path.exists(file_path):
                            with st.spinner("🔄 重新处理文档..."):
                                success = process_document_for_knowledge_graph(
                                    file_path, driver, client, doc['source_name']
                                )
                                if success:
                                    st.rerun()
                with col2:
                    if st.button("🗑️ 从图谱移除", key=f"remove_{doc['source_name']}"):
                        # 从图谱移除逻辑
                        if driver:
                            with driver.session(database=NEO4J_CONFIG["database"]) as session:
                                session.run("""
                                MATCH (doc:Document {source_name: $source_name})-[r:CONTAINS]->(e:Entity)
                                DETACH DELETE doc, r, e
                                """, parameters={'source_name': doc['source_name']})
                            st.success(f"✅ 已从知识图谱移除 '{doc['source_name']}'")
                            # 更新会话状态
                            st.session_state.processed_docs = [
                                d for d in st.session_state.processed_docs 
                                if d['source_name'] != doc['source_name']
                            ]
                            st.rerun()
                        

def render_knowledge_graph_qa_page(driver, client):
    """知识图谱问答页面"""
    st.subheader("🧠 基于知识图谱的智能问答")
    st.markdown("""
    💡 **智能问答系统**：基于知识图谱的实体关系进行精准问答
    
    1. **精准检索**：从知识图谱中检索相关实体和关系
    2. **专业回答**：由LLM基于专业知识生成权威解答
    3. **透明可信**：显示信息来源和置信度
    4. **领域专注**：专注于宝石切工专业知识
    """)
    
    # 检查连接状态
    if not driver:
        st.error("❌ 无法连接 Neo4j 数据库。请检查配置。")
        return
    
    if not client:
        st.error("❌ DeepSeek API 未配置。请设置 API Key。")
        return
    
    # 检查是否有已处理的文档
    uploaded_docs = get_uploaded_documents()
    if not uploaded_docs:
        st.warning("⚠️ 尚未上传任何文档。请先到 '文档管理' 页面上传并处理 Word 文档。")
        return
    
    # 获取图谱统计
    stats = get_graph_statistics(driver)
    if stats:
        st.info(f"📊 当前知识图谱包含: {stats['entity_count']}个实体, {stats['relationship_count']}个关系")
    
    # 问答界面
    question = st.text_area("❓ 请输入您的问题", 
                          placeholder="例如：玫瑰花琢型的特点是什么？谁发明了明亮式切工？",
                          height=100)
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("🔍 智能问答", type="primary", key="kg_query_button")
    with col2:
        if st.button("💡 问题示例"):
            st.info("""
            🔹 玫瑰花琢型是谁发明的？
            🔹 明亮式切工有什么特点？
            🔹 葵花琢型和玫瑰花琢型有什么关系？
            🔹 哪种切工最适合红宝石？
            🔹 巴洛克式切工的历史起源是什么？
            """)
    
    if ask_button:
        if not question.strip():
            st.warning("⚠️ 请输入问题后再提交。")
        else:
            with st.spinner("⚡ 正在检索知识图谱并生成回答..."):
                response = query_knowledge_graph(question, driver, client)
                
                # 显示结果
                st.divider()
                st.markdown("### 🎯 专业回答")
                st.write(response)
                
                # 显示检索到的相关三元组
                st.divider()
                st.markdown("### 📚 相关知识三元组")
                with driver.session(database=NEO4J_CONFIG["database"]) as session:
                    result = session.run("""
                    MATCH (e:Entity)-[r:RELATIONSHIP]-(related:Entity)
                    WHERE e.name CONTAINS $keyword OR related.name CONTAINS $keyword
                    RETURN 
                        e.name as subject,
                        r.type as predicate,
                        related.name as object,
                        r.confidence as confidence
                    ORDER BY r.confidence DESC
                    LIMIT 5
                    """, keyword=question[:20])
                    
                    triples_found = False
                    for record in result:
                        triples_found = True
                        with st.expander(f"🔗 {record['subject']} {record['predicate']} {record['object']}"):
                            st.write(f"**置信度**: {record['confidence']:.2f}")

def render_graph_statistics_page(driver):
    """图谱统计页面"""
    st.subheader("📊 知识图谱统计信息")
    
    if not driver:
        st.error("❌ 无法连接 Neo4j 数据库。请检查配置。")
        return
    
    if st.button("🔄 刷新统计", type="primary"):
        st.rerun()
    
    stats = get_graph_statistics(driver)
    if stats:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("实体数量", stats["entity_count"])
        with col2:
            st.metric("关系数量", stats["relationship_count"])
        with col3:
            st.metric("最后更新", stats["last_updated"])
        
        st.markdown("### 📊 实体类型分布")
        if stats["entity_types"]:
            for item in stats["entity_types"]:
                st.write(f"- **{item['type']}**: {item['count']} 个实体")
        else:
            st.info("ℹ️ 暂无实体类型统计信息")
        
        # 可视化建议
        st.markdown("### 📈 可视化建议")
        st.info("""
        您可以通过 Neo4j Browser (http://localhost:7474) 查看更详细的知识图谱可视化：
        
        ```cypher
        // 查看所有实体和关系
        MATCH (n)-[r]-(m) RETURN n, r, m LIMIT 50
        
        // 查看特定实体的关系
        MATCH (e:Entity {name: "玫瑰花琢型"})-[r]-(related) RETURN e, r, related
        
        // 查看最核心的实体
        MATCH (e:Entity)-[r]-() 
        RETURN e.name as entity, count(r) as relationship_count
        ORDER BY relationship_count DESC
        LIMIT 10
        ```
        """)
    else:
        st.warning("⚠️ 无法获取图谱统计信息，请检查Neo4j连接状态。")

def render_basic_query_page(driver):
    """基础查询页面 - 从第二个文件中提取的琢型查询功能"""
    st.subheader("🔍 宝石切工数据库查询")
    st.markdown("""
    💎 **基础查询功能**：浏览和搜索宝石切工数据库中的详细信息
    
    1. **完整切工列表**：查看所有已录入的宝石切工
    2. **关键词搜索**：通过名称快速定位特定切工
    3. **详细信息展示**：查看每种切工的结构描述、优缺点、适用材质等
    4. **图片展示**：查看切工的示例图片
    """)
    
    if not driver:
        st.error("❌ 无法连接 Neo4j 数据库。请检查配置。")
        return
    
    with driver.session(database=NEO4J_CONFIG["database"]) as session:
        try:
            # 获取所有切工
            result = session.run("MATCH (g:GemCut) RETURN g.name AS name ORDER BY g.name")
            all_cuts = [record['name'] for record in result]
            
            if not all_cuts:
                st.warning("⚠️ 数据库中暂无数据！")
                st.info("💡 请先通过'文档管理'页面上传相关文档，或手动添加切工数据到Neo4j数据库。")
                return
            
            st.success(f"✅ 共找到 **{len(all_cuts)}** 种切工")
            
            # 搜索功能
            search_term = st.text_input("🔍 搜索切工名称", placeholder="输入关键词...")
            filtered_cuts = [c for c in all_cuts if search_term.lower() in c.lower()] if search_term else all_cuts
            
            if not filtered_cuts:
                st.info("未找到匹配的切工。")
                return
            
            # 选择切工
            selected_cut = st.selectbox("选择切工名称查看详情", filtered_cuts)
            
            if selected_cut:
                # 获取切工详细信息
                data = session.run("MATCH (g:GemCut {name: $name}) RETURN g", name=selected_cut).single()
                if data:
                    g = data['g']
                    st.divider()
                    
                    # 使用两列布局 - 左侧文本，右侧图片
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"### {g['name']}")
                        if g.get('english_name'): 
                            st.caption(f"English: {g['english_name']}")
                        st.write(f"**结构描述**: {g.get('structure_description', 'N/A')}")
                        st.write(f"**优点**: {g.get('advantages', 'N/A')}")
                        st.write(f"**缺点**: {g.get('disadvantages', 'N/A')}")
                        st.write(f"**适用材质**: {g.get('suitable_materials', 'N/A')}")
                        st.write(f"**历史**: {g.get('history', 'N/A')}")
                        
                        # 添加更多字段（如果存在）
                        additional_fields = ['facets_count', 'refractive_index', 'brilliance_rating', 'hardness']
                        for field in additional_fields:
                            if g.get(field):
                                field_name = field.replace('_', ' ').title()
                                st.write(f"**{field_name}**: {g.get(field)}")
                    
                    with col2:
                        # 修复后的图片展示逻辑 - 针对您的具体路径
                        image_files = g.get('image_files', [])
                        if image_files:
                            st.markdown("**📸 切工示例图片**")
                            for i, img_path in enumerate(image_files[:3]):  # 只显示前3张
                                try:
                                    # 修复1：定义您的实际图片目录路径
                                    base_image_dir = r"D:\zhuomian\2207051004\data\images"
                                    
                                    # 修复2：清理文件名，移除可能的路径前缀
                                    clean_filename = os.path.basename(img_path)
                                    
                                    # 修复3：构建完整的绝对路径
                                    full_path = os.path.join(base_image_dir, clean_filename)
                                    
                                    # 修复4：检查文件是否存在，并尝试替代方案
                                    if not os.path.exists(full_path):
                                        # 尝试直接使用数据库中的路径（如果已经是绝对路径）
                                        if os.path.isabs(img_path) and os.path.exists(img_path):
                                            full_path = img_path
                                        else:
                                            # 尝试其他可能的路径组合
                                            possible_paths = [
                                                os.path.join(base_image_dir, img_path),
                                                os.path.join(os.path.dirname(base_image_dir), "images", clean_filename),
                                                os.path.join("D:\\zhuomian\\2207051004\\data", "images", clean_filename)
                                            ]
                                            
                                            found = False
                                            for path in possible_paths:
                                                if os.path.exists(path):
                                                    full_path = path
                                                    found = True
                                                    break
                                            
                                            if not found:
                                                st.warning(f"⚠️ 图片不存在: {clean_filename}")
                                                st.caption(f"尝试路径: {full_path}")
                                                continue
                                    
                                    # 修复5：使用正确的width参数，设置合适的图片大小
                                    st.image(full_path, caption=f"图 {i+1}", width='stretch')
                                    
                                except Exception as e:
                                    st.warning(f"图片加载失败: {os.path.basename(img_path)}")
                                    st.caption(f"错误详情: {str(e)}")
                                    st.caption(f"尝试路径: {full_path}")
                        else:
                            st.info("暂无图片")
        
        except Exception as e:
            st.error(f"❌ 查询数据库时出错：{e}")
            st.error(f"错误类型：{type(e).__name__}")
        
def render_ai_vision_page():
    """AI识图页面 - 直接集成，无需外部API"""
    st.subheader("📷 AI宝石切工智能识别")
    st.markdown("""
    🤖 **DINOv2 视觉识别引擎**：利用 Meta 最新的自监督视觉模型，精准识别宝石刻面结构与光影特征
    
    1. **上传图片**：上传宝石切工的清晰照片
    2. **特征提取**：使用DINOv2模型提取图片的深度特征
    3. **相似度匹配**：与预计算的特征库进行余弦相似度计算
    4. **专业识别**：返回最匹配的切工类型及其详细信息
    """)
    
    # 🔧 自动检查并生成特征库
    with st.spinner("🔧 检查特征库状态..."):
        from generate_features import generate_feature_db
        feature_db_ready = generate_feature_db()
    
    if not feature_db_ready:
        st.error("❌ 特征库准备失败，请检查上述错误信息")
        st.stop()
    
    # 1. 加载特征库
    feature_db_raw = load_feature_db()
    
    if feature_db_raw is None:
        st.error(f"❌ 加载特征库文件 `{AI_VISION_CONFIG['FEATURES_FILE']}` 失败")
        st.info("💡 请检查文件权限或重新生成特征库")
        st.stop()
    
    # 🔥 修复后的数据格式化：使用统一的图片路径
    feature_db = []
    if isinstance(feature_db_raw, dict):
        # 如果是字典 {'filename': vector}
        for filename, vector in feature_db_raw.items():
            # 获取纯净的文件名（移除路径前缀）
            clean_filename = os.path.basename(filename)
            
            # 使用统一的图片基础目录构建完整路径
            file_path = os.path.join(AI_VISION_CONFIG["IMAGE_BASE_DIR"], clean_filename)
            
            # 清理名称用于显示
            clean_name = os.path.splitext(clean_filename)[0].replace('_', ' ').title()
            
            feature_db.append({
                'name': clean_name,
                'vector': vector if isinstance(vector, np.ndarray) else np.array(vector),
                'path': file_path,
                'filename': clean_filename
            })
    elif isinstance(feature_db_raw, list):
        # 如果已经是列表格式，修复路径
        for item in feature_db_raw:
            # 获取纯净的文件名
            clean_filename = os.path.basename(item.get('filename', item.get('name', '')))
            
            # 重新构建正确路径
            file_path = os.path.join(AI_VISION_CONFIG["IMAGE_BASE_DIR"], clean_filename)
            
            # 清理名称
            clean_name = os.path.splitext(clean_filename)[0].replace('_', ' ').title()
            
            feature_db.append({
                'name': clean_name,
                'vector': item['vector'] if isinstance(item['vector'], np.ndarray) else np.array(item['vector']),
                'path': file_path,
                'filename': clean_filename
            })
    
    st.success(f"📚 特征库已就绪：包含 **{len(feature_db)}** 个切工样本")
    
    # 模型加载状态
    model_status = st.empty()
    model_status.info("⏳ 正在加载 DINOv2 视觉模型...")
    
    # 2. 加载模型
    model, transform, device = load_dinov2_model()
    
    if model is None:
        model_status.error("❌ 模型加载失败，请检查配置。")
        st.stop()
    
    model_status.success(f"✅ DINOv2 模型加载成功！设备：{device.upper()}")
    
    # 3. 上传图片
    st.markdown("### 📤 上传宝石图片")
    uploaded_file = st.file_uploader("请上传一张宝石切工图片", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(uploaded_file, caption="📸 您上传的图片", width='stretch')
        
        with st.spinner("⚡ 正在提取特征并比对 (DINOv2 引擎)..."):
            # 4. 提取上传图特征
            img_pil = Image.open(uploaded_file)
            query_embedding = get_image_embedding(img_pil, model, transform, device)
            
            if query_embedding is None:
                st.error("❌ 图片特征提取失败。")
                st.stop()
            
            # 5. 快速矩阵比对
            db_vectors = np.array([item['vector'] for item in feature_db])
            
            # 确保维度正确 (N, D)
            if len(db_vectors.shape) == 1:
                db_vectors = db_vectors.reshape(1, -1)
            
            # 计算余弦相似度
            similarities = cosine_similarity([query_embedding], db_vectors)[0]
            
            # 获取 Top 3 索引
            top_indices = np.argsort(similarities)[::-1][:3]
            
            # 6. 展示结果
            st.divider()
            st.markdown("### 🎯 识别结果")
            
            best_idx = top_indices[0]
            best_score = similarities[best_idx]
            best_match = feature_db[best_idx]
            
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                # 显示最佳匹配图片
                if os.path.exists(best_match['path']):
                    st.image(best_match['path'], 
                            caption=f"🏆 {best_match['name']}\n匹配度：{best_score:.2%}", 
                            width='stretch')
                else:
                    st.warning(f"匹配到的图片文件丢失：{best_match['path']}")
                    st.write(f"**名称**: {best_match['name']}")
            
            with res_col2:
                st.success(f"最相似的切工是：**{best_match['name']}**")
                st.metric(label="置信度得分", value=f"{best_score:.4f}")
                
                if best_score < 0.35:
                    st.warning("⚠️ 相似度较低，可能是拍摄角度特殊、光线过暗或该琢型不在数据库中。")
                elif best_score > 0.85:
                    st.balloons()
                    st.write("✨ **极高置信度匹配！**")
                elif best_score > 0.6:
                    st.write("✅ **可靠匹配**。")
                            
            # 显示 Top 2-3 的其他可能
            if len(top_indices) > 1:
                st.markdown("#### 📋 其他可能的匹配:")
                cols = st.columns(min(len(top_indices)-1, 3))
                for i, idx in enumerate(top_indices[1:3]):  # 只显示前2个其他匹配
                    if i >= len(cols):
                        break
                    item = feature_db[idx]
                    score = similarities[idx]
                    with cols[i]:
                        if os.path.exists(item['path']):
                            st.image(item['path'], 
                                    caption=f"{item['name']}\n{score:.2%}", 
                                    width=400)
                        else:
                            st.write(f"**{item['name']}**\n{score:.2%}")
    
    # 📝 使用说明
    st.divider()
    st.markdown("### ℹ️ 使用说明")
    st.info("""
    **图片要求**：
    - 清晰的宝石切工照片
    - 光线充足，避免反光
    - 正面视角最佳
    - 支持 JPG、PNG 格式
    
    
    **注意事项**：
    - 首次使用需要联网下载DINOv2模型
    - 特征库会自动在首次运行时生成
    - 确保 images/ 文件夹中有对应的参考图片
    """)

def render_blueprint_library_page():
    """琢型图纸库页面 - 从第二个文件中提取的琢型图纸库功能"""
    st.subheader("🔍 宝石琢型图纸检索系统")
    st.markdown("""
    📐 **专业图纸检索**：按折射率范围精确查找宝石琢型图纸
    
    1. **折射率分类**：按宝石折射率范围分类（RI=1.50-1.60, RI=1.60-1.70等）
    2. **琢型选择**：在选定折射率范围内选择具体琢型
    3. **图纸展示**：查看高清琢型图纸，支持分页浏览
    4. **专业参考**：为宝石切磨师提供精确的图纸参考
    """)
    
    # 📁 检查图纸根目录是否存在
    blueprints_root = BLUEPRINTS_CONFIG["root_dir"]
    if not os.path.exists(blueprints_root):
        st.error(f"❌ 错误：未找到图纸根目录 `{blueprints_root}`。请确保它与 `app.py` 在同一级目录下。")
        st.stop()
    
    # 🔍 第一步：获取所有折射率范围文件夹
    ri_folders = get_refractive_index_folders()
    
    if not ri_folders:
        st.warning(f"⚠️ 在 `{blueprints_root}` 中未找到符合 'RI=数字-数字' 格式的文件夹。")
        st.stop()
    
    # 📌 用户选择折射率范围
    selected_ri = st.selectbox(
        "🔹 步骤 1：选择折射率范围",
        ri_folders,
        key="ri_select"
    )
    
    if selected_ri:
        # 🔍 第二步：获取该折射率下的所有琢型文件夹
        cut_type_folders = get_cut_type_folders(selected_ri)
        
        if not cut_type_folders:
            st.info(f"📁 在 `{selected_ri}` 中未找到任何琢型文件夹。")
            st.stop()
        
        # 📌 用户选择琢型
        selected_cut = st.selectbox(
            "🔹 步骤 2：选择琢型",
            cut_type_folders,
            key="cut_select"
        )
        
        if selected_cut:
            # 🔍 第三步：获取该琢型下的所有图片文件
            image_files = get_blueprint_images(selected_ri, selected_cut)
            
            if not image_files:
                st.info(f"🖼️ `{selected_cut}` 文件夹中暂无图纸文件。")
                st.stop()
            
            total_images = len(image_files)
            st.success(f"✅ 已定位到：**{selected_ri} → {selected_cut}**，共 {total_images} 张图纸。")
            
            # 📄 分页控制
            col1, col2 = st.columns([2, 1])
            with col1:
                images_per_page = st.slider(
                    "每页显示数量",
                    min_value=3,
                    max_value=12,
                    value=6,
                    help="减少每页数量可显著提升加载速度"
                )
            with col2:
                st.write("")  # 留空占位
                st.caption(f"总图数：{total_images}")
            
            total_pages = (total_images + images_per_page - 1) // images_per_page
            page_number = st.selectbox(
                "📌 当前页",
                options=list(range(1, total_pages + 1)),
                format_func=lambda x: f"第 {x} 页 / 共 {total_pages} 页",
                key="page_select"
            )
            
            # 🖼️ 展示当前页图片
            start_idx = (page_number - 1) * images_per_page
            end_idx = min(start_idx + images_per_page, total_images)
            current_page_files = image_files[start_idx:end_idx]
            
            # 动态计算列数
            cols_count = min(3, len(current_page_files))
            if cols_count > 0:
                cols = st.columns(cols_count)
                for idx, img_file in enumerate(current_page_files):
                    img_full_path = os.path.join(blueprints_root, selected_ri, selected_cut, img_file)
                    with cols[idx % cols_count]:
                        try:
                            # 使用 width 参数限制尺寸，进一步加速渲染
                            pil_img = Image.open(img_full_path)
                            pil_img.thumbnail((220, 220))
                            st.image(
                                img_full_path,
                                caption=img_file
                            )
                        except Exception as e:
                            st.error(f"❌ 加载 `{img_file}` 失败：{str(e)[:50]}...")
            else:
                st.info("当前页无图片。")
            
            # 📊 底部状态栏
            st.caption(
                f"📊 显示范围：第 `{start_idx + 1}` ~ `{end_idx}` 张 | "
                f"当前页：{len(current_page_files)} 张"
            )

# ================= Streamlit 界面 =================
def main_menu():
    """主菜单界面"""
    st.sidebar.title("💎 宝石切工智能系统")
    st.sidebar.markdown("---")
    
    # 导航菜单
    menu_options = [
        "🏠 首页",
        "📚 文档管理", 
        "🧠 知识图谱问答",
        "📊 图谱统计",
        "🔍 基础查询",
        "📷 AI识图",
        "🔍 琢型图纸库"
    ]
    choice = st.sidebar.selectbox("导航菜单", menu_options, index=0)
    
    # 显示当前配置
    with st.sidebar.expander("⚙️ 系统配置", expanded=False):
        st.json({
            "Neo4j_URI": NEO4J_CONFIG["uri"],
            "LLM_模型": DEEPSEEK_CONFIG["model"],
            "AI识图模型": {
                "FEATURES_FILE": "features.pkl",
                "model_name": "dinov2_vits14"
            },
            "文档目录": APP_CONFIG["document_dir"],
            "图纸目录": BLUEPRINTS_CONFIG["root_dir"]
        })
        
    return choice

# ================= 主程序 =================
def main():
    """主程序入口"""
    st.set_page_config(
        page_title="💎 宝石切工智能系统",
        page_icon="💎",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 初始化会话状态
    if 'processed_docs' not in st.session_state:
        st.session_state.processed_docs = []
    
    # 初始化连接
    driver = get_neo4j_driver()
    client = init_deepseek_client()
    
    # 主菜单
    choice = main_menu()
    
    # 页面路由
    if choice == "🏠 首页":
        render_home_page()
    elif choice == "📚 文档管理":
        render_document_management_page(driver, client)
    elif choice == "🧠 知识图谱问答":
        render_knowledge_graph_qa_page(driver, client)
    elif choice == "📊 图谱统计":
        render_graph_statistics_page(driver)
    elif choice == "🔍 基础查询":
        render_basic_query_page(driver)
    elif choice == "📷 AI识图":
        render_ai_vision_page()
    elif choice == "🔍 琢型图纸库":
        render_blueprint_library_page()
    
    # 页脚
    st.markdown("---")
    st.markdown("💎 **宝石切工智能系统** | 基于LLM、Neo4j、DINOv2构建 | © 2026")

if __name__ == "__main__":
    main()
