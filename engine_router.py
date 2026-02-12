"""
引擎C：自适应融合路由器
基于MLP动态计算语义流和因果流的融合权重
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
import re


class AdaptiveFusionRouter:
    """自适应融合路由器"""
    
    def __init__(self):
        # 改进的MLP：输入特征增加到5维
        # 输入特征：5维（has_numbers, node_type_onehot, semantic_score, content_length_ratio, error_keywords）
        # 输出：1维（融合权重alpha）
        self.mlp = nn.Sequential(
            nn.Linear(5, 16),  # 增加隐藏层大小
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # 输出0-1之间的权重
        )
        # 初始化权重（倾向于使用因果得分，因为因果得分更能反映真正的因果关系）
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重：倾向于使用因果得分（alpha较小，即更依赖因果得分）"""
        with torch.no_grad():
            # 初始化权重，使输出倾向于较小的alpha值（更依赖因果得分）
            # 通过设置负偏置，使Sigmoid输出更接近0（更依赖因果得分）
            self.mlp[4].bias.fill_(-1.0)  # Sigmoid(-1) ≈ 0.27，更倾向于使用因果得分
    
    def _extract_features(
        self,
        parent_content: str,
        child_content: str,
        child_node_type: str,
        semantic_score: float
    ) -> np.ndarray:
        """
        提取特征向量（改进版：添加更多特征）
        
        Returns:
            特征向量 [has_numbers, node_type_onehot, semantic_score, content_length_ratio, error_keywords]
        """
        # 特征1：是否包含数值（0或1）
        has_numbers = 1.0 if bool(re.search(r'\d+', parent_content + child_content)) else 0.0
        
        # 特征2：子节点类型One-hot编码
        # 简化处理：Critic=1, 其他=0
        node_type_onehot = 1.0 if child_node_type == "Review" else 0.0
        
        # 特征3：语义得分（引擎A的输出）
        semantic_feature = semantic_score
        
        # 特征4：内容长度比例（归一化）
        parent_len = len(parent_content)
        child_len = len(child_content)
        total_len = parent_len + child_len
        content_length_ratio = parent_len / total_len if total_len > 0 else 0.5
        
        # 特征5：是否包含错误关键词（如"应该"、"错误"等可能表示不确定性的词）
        error_keywords = ['应该', '可能', '错误', '不对', '不正确', 'should', 'might', 'wrong', 'error']
        has_error_keyword = 1.0 if any(keyword in parent_content.lower() or keyword in child_content.lower() 
                                      for keyword in error_keywords) else 0.0
        
        return np.array([has_numbers, node_type_onehot, semantic_feature, content_length_ratio, has_error_keyword], dtype=np.float32)
    
    def compute_fusion_weight(
        self,
        parent_content: str,
        child_content: str,
        child_node_type: str,
        semantic_score: float,
        causal_score: float = None  # 可选：如果提供因果得分，可以用于更智能的融合
    ) -> float:
        """
        计算融合权重alpha
        
        Args:
            parent_content: 父节点内容
            child_content: 子节点内容
            child_node_type: 子节点类型
            semantic_score: 语义得分（引擎A的输出）
            causal_score: 因果得分（引擎B的输出），如果提供则用于改进权重计算
        
        Returns:
            融合权重alpha（0-1之间，越小越依赖因果得分）
        """
        # 提取特征
        features = self._extract_features(
            parent_content, child_content, child_node_type, semantic_score
        )
        
        # 如果提供了因果得分，可以调整特征
        # 如果因果得分很高，说明因果关系强，应该更依赖因果得分（alpha应该更小）
        if causal_score is not None:
            # 将因果得分作为额外信息：高因果得分 -> 更小的alpha（更依赖因果得分）
            # 这里我们通过调整特征来影响MLP的输出
            # 如果因果得分高，我们可以在特征中添加一个信号
            pass  # 当前实现中，MLP已经通过初始化倾向于使用因果得分
        
        # 转换为tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        # 前向传播
        with torch.no_grad():
            alpha = self.mlp(features_tensor).item()
        
        # 如果因果得分很高，进一步降低alpha（更依赖因果得分）
        if causal_score is not None and causal_score > 0.6:
            alpha = alpha * 0.7  # 降低alpha，更依赖因果得分
        
        return max(0.0, min(1.0, alpha))
    
    def fuse_scores(
        self,
        semantic_score: float,
        causal_score: float,
        alpha: float
    ) -> float:
        """
        融合语义得分和因果得分
        
        Args:
            semantic_score: 语义得分
            causal_score: 因果得分
            alpha: 融合权重
        
        Returns:
            融合后的边权重
        """
        # 公式：w = alpha * semantic_score + (1 - alpha) * causal_score
        fused_score = alpha * semantic_score + (1 - alpha) * causal_score
        return max(0.0, min(1.0, fused_score))
