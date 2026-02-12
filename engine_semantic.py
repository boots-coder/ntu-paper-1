"""
引擎A：语义注意力追踪器
基于DistilBERT的Attention矩阵计算语义相似度
"""
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from typing import Tuple
import numpy as np


class SemanticAttentionTracer:
    """语义注意力追踪器"""
    
    def __init__(self):
        self.model_name = "distilbert-base-uncased"
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        # 使用eager attention实现以避免警告
        self.model = DistilBertModel.from_pretrained(
            self.model_name,
            attn_implementation="eager"
        )
        self.model.eval()  # 设置为评估模式
    
    def compute_semantic_score(self, parent_content: str, child_content: str) -> float:
        """
        计算父子节点间的语义注意力得分
        
        Args:
            parent_content: 父节点内容
            child_content: 子节点内容
        
        Returns:
            语义得分（0-1之间的浮点数）
        """
        try:
            # 使用tokenizer的encode_plus方法，自动添加[CLS]和[SEP]
            inputs = self.tokenizer(
                parent_content,
                child_content,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # 获取token位置信息
            input_ids = inputs["input_ids"][0]
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            
            # 找到[SEP]标记的位置
            sep_token_id = self.tokenizer.sep_token_id
            sep_positions = [i for i, token_id in enumerate(input_ids) if token_id == sep_token_id]
            
            if len(sep_positions) < 1:
                # 如果没有找到SEP，返回默认值
                return 0.5
            
            # parent: [CLS] ... [SEP]
            # child: [SEP] ... [SEP] (如果有第二个SEP)
            parent_start = 1  # 跳过[CLS]
            parent_end = sep_positions[0]
            child_start = sep_positions[0] + 1
            child_end = sep_positions[1] if len(sep_positions) > 1 else len(tokens)
            
            # 前向传播
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
                attentions = outputs.attentions  # 所有层的注意力矩阵
            
            # 使用最后一层的注意力矩阵
            last_layer_attention = attentions[-1]  # shape: (batch_size, num_heads, seq_len, seq_len)
            
            # 取所有头的平均值
            attention_matrix = last_layer_attention.mean(dim=1).squeeze(0)  # shape: (seq_len, seq_len)
            
            # 提取从child tokens指向parent tokens的注意力子矩阵
            # attention_matrix[i, j] 表示token i对token j的注意力
            # 我们关注child tokens对parent tokens的注意力
            if child_start < child_end and parent_end > parent_start:
                # 子节点token对父节点token的注意力
                child_to_parent_attention = attention_matrix[child_start:child_end, parent_start:parent_end]
                
                if child_to_parent_attention.numel() > 0:
                    # 沿子节点token维度取平均值（Mean Pooling）
                    semantic_score = child_to_parent_attention.mean().item()
                    # 归一化到0-1范围（注意力值通常在0-1之间，但为了保险起见）
                    semantic_score = max(0.0, min(1.0, semantic_score))
                    return semantic_score
        except Exception as e:
            print(f"语义引擎计算错误: {e}")
        
        # 如果无法计算，返回默认值
        return 0.5
