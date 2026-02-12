"""
引擎B：蒙特卡洛因果推断器
通过扰动变量，利用LLM计算真实的因果依赖度
"""
import re
import random
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from config import OPENAI_API_KEY, OPENAI_MODEL, MONTE_CARLO_SAMPLES, PERTURBATION_MASK_RATIO

# 尝试导入新版本的OpenAI客户端
try:
    from openai import OpenAI
    OPENAI_NEW_API = True
except ImportError:
    try:
        import openai
        OPENAI_NEW_API = False
    except ImportError:
        raise ImportError("请安装openai包: pip install openai")


class MonteCarloCausalAuditor:
    """蒙特卡洛因果推断器"""
    
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("请在config.py中设置OPENAI_API_KEY")
        
        self.model_name = OPENAI_MODEL
        self.sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.monte_carlo_samples = MONTE_CARLO_SAMPLES
        
        # 初始化OpenAI客户端
        if OPENAI_NEW_API:
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        else:
            openai.api_key = OPENAI_API_KEY
            self.client = None
    
    def _perturb_node(self, content: str) -> str:
        """
        对节点内容进行扰动
        
        扰动策略：
        1. 如果包含数字，将其±10%或替换为随机数
        2. 如果不包含数字，随机Mask掉15%的名词/动词
        """
        # 检查是否包含数字
        numbers = re.findall(r'\d+\.?\d*', content)
        
        if numbers:
            # 策略1：修改数字
            perturbed = content
            for num_str in numbers:
                try:
                    num = float(num_str)
                    # 随机选择：±10%或替换为随机数
                    if random.random() < 0.5:
                        # ±10%
                        perturbation = num * random.uniform(0.9, 1.1)
                    else:
                        # 替换为随机数（在合理范围内）
                        perturbation = random.uniform(0, num * 2) if num > 0 else random.uniform(num * 2, 0)
                    
                    perturbed = perturbed.replace(num_str, str(int(perturbation) if num.is_integer() else round(perturbation, 2)), 1)
                except:
                    pass
            return perturbed
        else:
            # 策略2：Mask掉部分词汇
            words = content.split()
            mask_count = max(1, int(len(words) * PERTURBATION_MASK_RATIO))
            indices_to_mask = random.sample(range(len(words)), min(mask_count, len(words)))
            
            perturbed_words = words.copy()
            for idx in indices_to_mask:
                perturbed_words[idx] = "[MASK]"
            
            return " ".join(perturbed_words)
    
    def _generate_counterfactual(self, context: str, parent_content: str, agent_role: str) -> str:
        """
        使用LLM生成反事实响应
        
        Args:
            context: 背景上下文
            parent_content: 扰动后的父节点内容
            agent_role: 智能体角色
        
        Returns:
            生成的响应内容
        """
        # 优化的prompt：更明确地要求基于给定前提生成响应，强调因果依赖
        prompt = f"""你正在分析一个多智能体系统的执行轨迹。当前背景上下文是：

{context}

现在，假设前提条件发生了变化：
前提：{parent_content}

请严格基于这个变化后的前提，扮演{agent_role}角色，生成下一步的响应。注意：
1. 必须基于给定的前提条件进行推理
2. 如果前提中的关键信息发生变化，响应应该相应改变
3. 只输出响应内容，不要添加任何解释或说明

响应："""
        
        try:
            if OPENAI_NEW_API:
                # 新版本API
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "你是一个多智能体系统中的智能体。你的任务是分析前提条件的变化如何影响后续响应。如果前提改变，响应必须相应改变以反映这种因果关系。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.8,  # 稍微提高温度以获得更多样化的反事实响应
                    max_tokens=200
                )
                return response.choices[0].message.content.strip()
            else:
                # 旧版本API
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "你是一个多智能体系统中的智能体。你的任务是分析前提条件的变化如何影响后续响应。如果前提改变，响应必须相应改变以反映这种因果关系。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.8,  # 稍微提高温度以获得更多样化的反事实响应
                    max_tokens=200
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM调用错误: {e}")
            return parent_content  # 失败时返回原内容
    
    def _compute_semantic_distance(self, text1: str, text2: str) -> float:
        """
        计算两个文本的语义距离（余弦距离）
        
        Returns:
            距离值（0-1之间，0表示完全相同，1表示完全不同）
        """
        embeddings = self.sentence_model.encode([text1, text2])
        # 计算余弦相似度
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        # 转换为距离（1 - 相似度）
        distance = 1 - similarity
        return max(0.0, min(1.0, distance))
    
    def compute_causal_score(
        self,
        parent_content: str,
        child_content: str,
        context: str,
        agent_role: str
    ) -> float:
        """
        计算因果得分
        
        Args:
            parent_content: 父节点内容
            child_content: 子节点内容（真实的）
            context: 背景上下文
            agent_role: 智能体角色
        
        Returns:
            因果得分（0-1之间的浮点数）
        """
        # 生成N个扰动版本
        perturbed_parents = [self._perturb_node(parent_content) for _ in range(self.monte_carlo_samples)]
        
        # 生成反事实响应
        counterfactual_children = []
        for perturbed_parent in perturbed_parents:
            counterfactual = self._generate_counterfactual(context, perturbed_parent, agent_role)
            counterfactual_children.append(counterfactual)
        
        # 计算真实子节点与所有反事实子节点的语义距离
        distances = []
        for counterfactual_child in counterfactual_children:
            distance = self._compute_semantic_distance(child_content, counterfactual_child)
            distances.append(distance)
        
        # 计算平均距离作为因果得分
        # 距离越大，说明父节点变化对子节点影响越大，因果性越强
        causal_score = np.mean(distances)
        
        return max(0.0, min(1.0, causal_score))
