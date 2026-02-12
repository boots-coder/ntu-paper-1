"""
配置文件：用于存储API密钥和其他配置信息
请在此文件中填入您的OpenAI API密钥
"""
import os

# OpenAI API配置
# 请设置环境变量 OPENAI_API_KEY，或在本地配置文件中设置（不要提交到git）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o"  # 使用GPT-4o模型

# 其他配置
DAMPING_FACTOR = 0.4  # 阻尼系数（进一步降低以让能量更充分向上传播到真正的根因）
MONTE_CARLO_SAMPLES = 10  # 蒙特卡洛采样数量（增加以提高因果推断准确性）
PERTURBATION_MASK_RATIO = 0.15  # 扰动时mask的比例

# 模型配置
DISTILBERT_MODEL_NAME = "distilbert-base-uncased"
SENTENCE_BERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
