"""
阻尼反向传播算法
实现马尔可夫能量衰减传播，计算每个节点的责任分数
"""
from typing import Dict, List, Tuple, Optional
from graph_builder import GraphBuilder
from config import DAMPING_FACTOR


class DampedBackwardPropagation:
    """阻尼反向传播算法"""
    
    def __init__(self, graph_builder: GraphBuilder, edge_weights: Dict[Tuple[str, str], float]):
        """
        Args:
            graph_builder: 图构建器
            edge_weights: 边权重字典，key为(parent_id, child_id)的元组
        """
        self.graph_builder = graph_builder
        self.edge_weights = edge_weights
        self.damping_factor = DAMPING_FACTOR
        self.blame_scores: Dict[str, float] = {}
    
    def propagate(self, error_sink_node_id: str) -> Dict[str, float]:
        """
        执行阻尼反向传播
        
        Args:
            error_sink_node_id: 错误汇点节点ID
        
        Returns:
            每个节点的责任分数字典
        """
        # 初始化：所有节点责任分数为0
        for node in self.graph_builder.trace.nodes:
            self.blame_scores[node.node_id] = 0.0
        
        # 注入初始错误能量
        self.blame_scores[error_sink_node_id] = 1.0
        
        # 获取逆拓扑排序序列
        reverse_topological_order = self.graph_builder.get_reverse_topological_order()
        
        if not reverse_topological_order:
            print("警告：无法进行拓扑排序，返回初始分数")
            return self.blame_scores
        
        # 传播主循环
        for node_id in reverse_topological_order:
            current_blame = self.blame_scores.get(node_id, 0.0)
            
            if current_blame <= 0:
                continue  # 跳过没有责任的节点
            
            # 获取所有父节点
            parent_ids = self.graph_builder.get_parents(node_id)
            
            if not parent_ids:
                # 如果没有父节点，责任全部截留在当前节点
                continue
            
            # 计算当前节点需向上分配的总能量
            # E_up = (1 - damping) * E_current
            energy_to_distribute = (1 - self.damping_factor) * current_blame
            
            # 剩下的能量截留在当前节点自身
            self.blame_scores[node_id] = self.damping_factor * current_blame
            
            # 计算局部归一化权重分母
            # Z = sum(w(parent, current))
            normalization_sum = 0.0
            for parent_id in parent_ids:
                edge_key = (parent_id, node_id)
                if edge_key in self.edge_weights:
                    normalization_sum += self.edge_weights[edge_key]
            
            # 按权重比例回传给父节点
            # 改进：使用平方权重来放大高权重边的影响，使真正的因果链更明显
            if normalization_sum > 0:
                # 计算平方权重和（用于归一化）
                squared_weights_sum = sum(
                    (self.edge_weights.get((parent_id, node_id), 0.0) ** 2)
                    for parent_id in parent_ids
                )
                
                for parent_id in parent_ids:
                    edge_key = (parent_id, node_id)
                    if edge_key in self.edge_weights:
                        weight = self.edge_weights[edge_key]
                        # 使用平方权重来放大高权重边的影响
                        # 这样可以更好地识别真正的因果链
                        if squared_weights_sum > 0:
                            squared_weight = weight ** 2
                            parent_energy = (squared_weight / squared_weights_sum) * energy_to_distribute
                        else:
                            parent_energy = (weight / normalization_sum) * energy_to_distribute
                        self.blame_scores[parent_id] = self.blame_scores.get(parent_id, 0.0) + parent_energy
                    else:
                        # 逻辑彻底断链的极端兜底：均分
                        parent_energy = energy_to_distribute / len(parent_ids)
                        self.blame_scores[parent_id] = self.blame_scores.get(parent_id, 0.0) + parent_energy
            else:
                # 如果所有边的权重都是0，均分给所有父节点
                parent_energy = energy_to_distribute / len(parent_ids)
                for parent_id in parent_ids:
                    self.blame_scores[parent_id] = self.blame_scores.get(parent_id, 0.0) + parent_energy
        
        return self.blame_scores
    
    def get_root_cause(self, error_sink_node_id: str = None, error_probs: Dict[str, float] = None) -> Tuple[str, float]:
        """
        获取根因节点（结合错误概率）
        
        Args:
            error_sink_node_id: 错误汇点节点ID，如果提供则排除它作为根因候选
            error_probs: 错误概率字典 {node_id: P(Error|v_i)}，如果提供则计算最终分数
        
        Returns:
            (node_id, final_score) 元组
        """
        if not self.blame_scores:
            return ("", 0.0)
        
        # 计算最终分数：Final_Score(v_i) = B_propagated(v_i) × P(Error|v_i)
        if error_probs:
            final_scores = {
                node_id: score * error_probs.get(node_id, 0.5)
                for node_id, score in self.blame_scores.items()
            }
        else:
            final_scores = self.blame_scores
        
        # 排除错误汇点作为根因候选
        candidates = {
            node_id: score 
            for node_id, score in final_scores.items() 
            if error_sink_node_id is None or node_id != error_sink_node_id
        }
        
        if not candidates:
            # 如果没有候选节点，返回错误汇点
            root_cause_id = max(final_scores.items(), key=lambda x: x[1])[0]
            return (root_cause_id, final_scores[root_cause_id])
        
        root_cause_id = max(candidates.items(), key=lambda x: x[1])[0]
        return (root_cause_id, candidates[root_cause_id])
