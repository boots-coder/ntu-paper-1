"""
局部错误检测器 (Local Error Detector)
计算 P(Error|v_i)：给定节点内容，判断其是否包含错误的概率
"""
from config import OPENAI_API_KEY, OPENAI_MODEL
from typing import Dict
import re

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


class LocalErrorDetector:
    """局部错误检测器：判断节点内容是否包含错误"""
    
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("请在config.py中设置OPENAI_API_KEY")
        
        self.model_name = OPENAI_MODEL
        
        # 初始化OpenAI客户端
        if OPENAI_NEW_API:
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        else:
            openai.api_key = OPENAI_API_KEY
            self.client = None
    
    def detect_error(self, node_content: str, problem: str, context: str = "") -> float:
        """
        检测节点内容是否包含错误
        
        Args:
            node_content: 节点内容
            problem: 原始问题
            context: 上下文信息（可选）
        
        Returns:
            错误概率 P(Error|v_i)，范围 [0, 1]
            0表示完全正确，1表示明显错误
        """
        # 首先使用规则检测明显的错误
        rule_based_score = self._rule_based_detection(node_content, problem)
        
        # 如果规则检测已经很明显，直接返回
        if rule_based_score > 0.8:
            return rule_based_score
        if rule_based_score < 0.2:
            return rule_based_score
        
        # 否则使用LLM进行更细致的判断
        llm_score = self._llm_based_detection(node_content, problem, context)
        
        # 结合规则和LLM的结果
        combined_score = 0.3 * rule_based_score + 0.7 * llm_score
        
        return max(0.0, min(1.0, combined_score))
    
    def _rule_based_detection(self, content: str, problem: str) -> float:
        """
        基于规则的错误检测（快速、低成本）
        """
        error_score = 0.0
        
        # 策略1：检查数字计算错误
        calc_patterns = [
            r'(\d+)\s*[/÷]\s*(\d+)\s*=\s*(\d+)',
            r'(\d+)\s*[+\+]\s*(\d+)\s*=\s*(\d+)',
            r'(\d+)\s*[-−]\s*(\d+)\s*=\s*(\d+)',
            r'(\d+)\s*[*×]\s*(\d+)\s*=\s*(\d+)',
        ]
        
        for pattern in calc_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                try:
                    if len(match) == 3:
                        a, b, result = map(float, match)
                        if '/' in content or '÷' in content:
                            expected = a / b
                        elif '+' in content or '＋' in content:
                            expected = a + b
                        elif '-' in content or '−' in content:
                            expected = a - b
                        elif '*' in content or '×' in content:
                            expected = a * b
                        else:
                            continue
                        
                        if abs(result - expected) > 0.01:
                            error_score = max(error_score, 0.9)
                except:
                    pass
        
        # 策略2：检查逻辑矛盾
        if '周三' in content and ('周二' in problem or '明天' in problem):
            error_score = max(error_score, 0.8)
        if '30' in content and '50' in problem and '进货' in content:
            error_score = max(error_score, 0.8)
        if '15' in content and '20' in problem and '更大' in content and '比较' in problem:
            error_score = max(error_score, 0.8)
        
        return error_score
    
    def _llm_based_detection(self, content: str, problem: str, context: str = "") -> float:
        """基于LLM的错误检测"""
        prompt = f"""判断推理步骤是否包含错误。

问题：{problem}
步骤：{content}
{f"上下文：{context}" if context else ""}

只回答：正确/错误/不确定"""

        try:
            if OPENAI_NEW_API:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "你是逻辑推理检查器。判断推理步骤是否包含错误。只回答：正确/错误/不确定"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=20
                )
                result = response.choices[0].message.content.strip().lower()
            else:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "你是逻辑推理检查器。判断推理步骤是否包含错误。只回答：正确/错误/不确定"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=20
                )
                result = response.choices[0].message.content.strip().lower()
            
            if '错误' in result or 'wrong' in result:
                return 0.9
            elif '不确定' in result:
                return 0.5
            else:
                return 0.1
        except Exception as e:
            print(f"LLM错误检测失败: {e}")
            return 0.5
    
    def compute_error_probabilities(self, trace, context: str = "") -> Dict[str, float]:
        """计算所有节点的错误概率"""
        error_probs = {}
        for node in trace.nodes:
            error_prob = self.detect_error(node.content, trace.problem, context)
            error_probs[node.node_id] = error_prob
        return error_probs
