import os
import torch
import pickle
from PIL import Image
from torchvision import transforms
import numpy as np

def debug_image_path():
    """调试图片路径问题"""
    print("="*50)
    print(f"当前工作目录: {os.getcwd()}")
    print(f"images 路径: {os.path.abspath('images')}")
    
    # 检查 images 目录是否存在
    if not os.path.exists('images'):
        print("❌ images 目录不存在！")
        print(f"尝试路径: {os.path.abspath('images')}")
    else:
        print("✅ images 目录存在")
        
        # 检查是否有图片文件
        files = os.listdir('images')
        print(f"images 中的文件数量: {len(files)}")
        print(f"文件列表: {files[:5]}...")  # 只显示前5个
        
        # 检查 round.jpg 是否存在
        if 'round.jpg' in files:
            print("✅ round.jpg 存在！")
        else:
            print("❌ round.jpg 不存在！")
        
        # 检查文件名是否纯字母
        print("\n检查文件名规则:")
        for f in files[:10]:
            name, ext = os.path.splitext(f)
            print(f"  - {f}: name='{name}', isalpha={name.isalpha()}")
    
    print("="*50)
    
def generate_feature_db():
    """动态生成特征库，如果文件不存在则自动创建"""
    
    # ✅ 关键修复1：使用绝对路径保存特征库
    current_dir = os.path.dirname(os.path.abspath(__file__))
    IMAGE_DIR = os.path.join(current_dir, "images")
    FEATURES_FILE = os.path.join(current_dir, "features.pkl")
    
    # ✅ 关键修复2：使用绝对路径访问图片
    IMAGE_DIR = os.path.join(current_dir, "images")  # 绝对路径
    
    print(f"📍 当前工作目录: {os.getcwd()}")
    print(f"📍 脚本所在目录: {current_dir}")
    print(f"📍 特征库保存路径: {FEATURES_FILE}")
    print(f"📍 图片目录路径: {IMAGE_DIR}")
    
    # 检查是否已存在特征库
    if os.path.exists(FEATURES_FILE):
        print(f"✅ 特征库文件 '{FEATURES_FILE}' 已存在，跳过生成")
        return True
    
    print("⏳ 特征库文件不存在，开始自动生成...")
    
    # 检查图片目录
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ 错误：找不到图片文件夹 '{IMAGE_DIR}'")
        return False
    
    # 加载DINOv2模型
    print("⏳ 正在加载 DINOv2 模型...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', 
                          pretrained=True, trust_repo=True)
    model.eval()
    print("✅ DINOv2 模型加载成功！")
    
    # 预处理配置
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    def extract_feature(img_path):
        """提取单张图片的特征向量"""
        img = Image.open(img_path).convert("RGB")
        input_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            features_dict = model.forward_features(input_tensor)
            cls_token = features_dict['x_norm_clstoken']
            feat = cls_token.squeeze(0).cpu().numpy().astype(np.float32)
        
        # 归一化
        norm = np.linalg.norm(feat, ord=2)
        return feat / norm if norm > 0 else feat
    
    def is_primary_image(filename):
        """判断是否为主图（纯字母文件名）"""
        name, ext = os.path.splitext(filename.lower())
        return ext in (".jpg", ".jpeg", ".png") and name.isalpha()
    
    # 提取特征
    features = {}
    image_count = 0
    
    print(f"📁 扫描图片文件夹: {IMAGE_DIR}")
   # ✅ 关键修改：存储绝对路径而不是文件名
    for filename in sorted(os.listdir(IMAGE_DIR)):
        filepath = os.path.join(IMAGE_DIR, filename)
        if not os.path.isfile(filepath):
            continue
            
        if is_primary_image(filename):
            try:
                feat = extract_feature(filepath)
                # 存储完整信息，包括绝对路径
                features[filename] = {
                    'feature': feat,
                    'path': filepath,  # ✅ 存储绝对路径
                    'name': filename.split('.')[0].capitalize()
                }
                image_count += 1
                print(f"✅ 提取: {filename} → 路径: {filepath}")
            except Exception as e:
                print(f"❌ 跳过 {filename}: {e}")
                
    # 保存特征库
    try:
        with open(FEATURES_FILE, "wb") as f:
            pickle.dump(features, f)
        print(f"🎉 特征库生成成功！保存到: {FEATURES_FILE}")
        print(f"   🔗 文件大小: {os.path.getsize(FEATURES_FILE)/1024:.1f}KB")
        
        # ✅ 关键修复3：验证文件是否真的存在
        if os.path.exists(FEATURES_FILE):
            print("✅ 验证成功：文件确实存在！")
        else:
            print("❌ 验证失败：文件不存在！")
    except Exception as e:
        print(f"❌ 保存特征库失败: {e}")
    
    return True
