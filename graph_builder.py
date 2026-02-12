"""
图构建模块：构建DAG并进行拓扑排序
"""
import networkx as nx
from typing import List, Dict, Tuple
from models import InputTrace, Node


class GraphBuilder:
    """图构建器"""
    
    def __init__(self, trace: InputTrace):
        self.trace = trace
        self.graph = nx.DiGraph()
        self.node_map: Dict[str, Node] = {}
        self._build_graph()
    
    def _build_graph(self):
        """构建有向无环图"""
        # 添加所有节点
        for node in self.trace.nodes:
            self.graph.add_node(node.node_id, node=node)
            self.node_map[node.node_id] = node
        
        # 添加所有边（根据parent_ids）
        for node in self.trace.nodes:
            for parent_id in node.parent_ids:
                if parent_id in self.node_map:
                    self.graph.add_edge(parent_id, node.node_id)
    
    def get_reverse_topological_order(self) -> List[str]:
        """获取逆拓扑排序序列（从结果向原因遍历）"""
        try:
            # 获取拓扑排序（从原因到结果）
            topological_order = list(nx.topological_sort(self.graph))
            # 反转得到逆拓扑排序（从结果到原因）
            return list(reversed(topological_order))
        except nx.NetworkXError as e:
            # 如果存在环，返回空列表
            print(f"警告：图中存在环，无法进行拓扑排序: {e}")
            return []
    
    def get_parents(self, node_id: str) -> List[str]:
        """获取节点的所有父节点"""
        return list(self.graph.predecessors(node_id))
    
    def get_children(self, node_id: str) -> List[str]:
        """获取节点的所有子节点"""
        return list(self.graph.successors(node_id))
    
    def has_cycle(self) -> bool:
        """检查图中是否存在环"""
        try:
            nx.find_cycle(self.graph)
            return True
        except nx.NetworkXNoCycle:
            return False
    
    def get_node(self, node_id: str) -> Node:
        """根据node_id获取节点对象"""
        return self.node_map.get(node_id)
