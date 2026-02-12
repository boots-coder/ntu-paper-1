"""
数据模型定义：用于解析输入和输出JSON
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class AgentRole(str, Enum):
    """智能体角色枚举"""
    PROPOSER = "Proposer"
    CRITIC = "Critic"
    TOOL = "Tool"
    SUMMARIZER = "Summarizer"


class NodeType(str, Enum):
    """节点类型枚举"""
    THOUGHT = "Thought"
    OBSERVATION = "Observation"
    REVIEW = "Review"
    OUTPUT = "Output"


@dataclass
class Node:
    """节点数据模型"""
    node_id: str
    agent_role: str
    node_type: str
    content: str
    parent_ids: List[str] = field(default_factory=list)


@dataclass
class InputTrace:
    """输入轨迹数据模型"""
    trace_id: str
    problem: str
    error_sink_node_id: str
    nodes: List[Node]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InputTrace':
        """从字典创建InputTrace对象"""
        nodes = [Node(**node_data) for node_data in data.get("nodes", [])]
        return cls(
            trace_id=data["trace_id"],
            problem=data["problem"],
            error_sink_node_id=data["error_sink_node_id"],
            nodes=nodes
        )


@dataclass
class BlameDistribution:
    """责任分布数据模型"""
    node_id: str
    agent_role: str
    node_type: str
    blame_score: float
    diagnosis: str


@dataclass
class DiagnosticResults:
    """诊断结果数据模型"""
    root_cause_node_id: str
    root_cause_agent_role: str
    blame_distribution: List[BlameDistribution]


@dataclass
class Metrics:
    """指标数据模型"""
    semantic_engine_invocations: int = 0
    causal_engine_invocations: int = 0
    monte_carlo_samples_generated: int = 0


@dataclass
class OutputResult:
    """输出结果数据模型"""
    trace_id: str
    status: str
    diagnostic_results: DiagnosticResults
    metrics: Metrics

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "trace_id": self.trace_id,
            "status": self.status,
            "diagnostic_results": {
                "root_cause_node_id": self.diagnostic_results.root_cause_node_id,
                "root_cause_agent_role": self.diagnostic_results.root_cause_agent_role,
                "blame_distribution": [
                    {
                        "node_id": bd.node_id,
                        "agent_role": bd.agent_role,
                        "node_type": bd.node_type,
                        "blame_score": float(round(bd.blame_score, 3)),  # 确保转换为Python float
                        "diagnosis": bd.diagnosis
                    }
                    for bd in self.diagnostic_results.blame_distribution
                ]
            },
            "metrics": {
                "semantic_engine_invocations": int(self.metrics.semantic_engine_invocations),
                "causal_engine_invocations": int(self.metrics.causal_engine_invocations),
                "monte_carlo_samples_generated": int(self.metrics.monte_carlo_samples_generated)
            }
        }
