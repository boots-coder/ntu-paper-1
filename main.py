"""
主流程：整合所有模块，实现完整的错误溯源系统
"""
import json
from typing import Dict, Tuple
from models import InputTrace, OutputResult, DiagnosticResults, BlameDistribution, Metrics
from graph_builder import GraphBuilder
from engine_semantic import SemanticAttentionTracer
from engine_causal import MonteCarloCausalAuditor
from engine_router import AdaptiveFusionRouter
from propagation import DampedBackwardPropagation
from error_detector import LocalErrorDetector


class ASCMASSystem:
    """ASC-MAS错误溯源系统"""
    
    def __init__(self):
        self.semantic_engine = SemanticAttentionTracer()
        self.causal_engine = MonteCarloCausalAuditor()
        self.router = AdaptiveFusionRouter()
        self.error_detector = LocalErrorDetector()  # 新增：错误检测器
        self.metrics = Metrics()
    
    def compute_edge_weights(self, graph_builder: GraphBuilder) -> Dict[Tuple[str, str], float]:
        """
        计算所有边的权重
        
        Args:
            graph_builder: 图构建器
        
        Returns:
            边权重字典，key为(parent_id, child_id)的元组
        """
        edge_weights = {}
        trace = graph_builder.trace
        
        # 构建上下文（所有节点的内容）
        context = trace.problem + "\n" + "\n".join([f"{node.node_id}: {node.content}" for node in trace.nodes])
        
        # 遍历所有边
        for child_node in trace.nodes:
            for parent_id in child_node.parent_ids:
                parent_node = graph_builder.get_node(parent_id)
                if not parent_node:
                    continue
                
                # 调用引擎A：语义注意力追踪器
                semantic_score = self.semantic_engine.compute_semantic_score(
                    parent_node.content,
                    child_node.content
                )
                self.metrics.semantic_engine_invocations += 1
                
                # 调用引擎B：蒙特卡洛因果推断器
                causal_score = self.causal_engine.compute_causal_score(
                    parent_node.content,
                    child_node.content,
                    context,
                    child_node.agent_role
                )
                self.metrics.causal_engine_invocations += 1
                self.metrics.monte_carlo_samples_generated += self.causal_engine.monte_carlo_samples
                
                # 调用引擎C：自适应融合路由器（传递因果得分以改进权重计算）
                alpha = self.router.compute_fusion_weight(
                    parent_node.content,
                    child_node.content,
                    child_node.node_type,
                    semantic_score,
                    causal_score=causal_score  # 传递因果得分以改进融合权重
                )
                
                # 融合得分
                fused_weight = self.router.fuse_scores(semantic_score, causal_score, alpha)
                
                edge_weights[(parent_id, child_node.node_id)] = fused_weight
        
        return edge_weights
    
    def generate_diagnosis(self, node_id: str, blame_score: float, root_cause_id: str) -> str:
        """
        生成诊断结论
        
        Args:
            node_id: 节点ID
            blame_score: 责任分数
            root_cause_id: 根因节点ID
        
        Returns:
            诊断结论字符串
        """
        if node_id == root_cause_id:
            return "首要责任节点 (Root Cause)。因果推断判定其对错误输出有强干预效应。"
        elif blame_score > 0.2:
            return "次要/连带责任节点。存在审查失职，未能阻断错误链路。"
        elif blame_score > 0:
            return "背景传导节点 (阻尼衰减截留部分分数)。"
        else:
            return "无责任。"
    
    def process(self, input_data: dict) -> OutputResult:
        """
        处理输入数据，返回诊断结果
        
        Args:
            input_data: 输入JSON字典
        
        Returns:
            输出结果对象
        """
        try:
            # 解析输入
            trace = InputTrace.from_dict(input_data)
            
            # 构建图
            graph_builder = GraphBuilder(trace)
            
            # 检查是否有环
            if graph_builder.has_cycle():
                return OutputResult(
                    trace_id=trace.trace_id,
                    status="error",
                    diagnostic_results=DiagnosticResults(
                        root_cause_node_id="",
                        root_cause_agent_role="",
                        blame_distribution=[]
                    ),
                    metrics=self.metrics
                )
            
            # 计算边权重
            edge_weights = self.compute_edge_weights(graph_builder)
            
            # 执行阻尼反向传播
            propagation = DampedBackwardPropagation(graph_builder, edge_weights)
            blame_scores = propagation.propagate(trace.error_sink_node_id)
            
            # 计算错误概率 P(Error|v_i)
            context = trace.problem + "\n" + "\n".join([f"{node.node_id}: {node.content}" for node in trace.nodes])
            error_probs = self.error_detector.compute_error_probabilities(trace, context)
            
            # 计算最终分数：Final_Score(v_i) = B_propagated(v_i) × P(Error|v_i)
            final_scores = {}
            for node_id, propagated_score in blame_scores.items():
                error_prob = error_probs.get(node_id, 0.5)  # 默认0.5（不确定）
                final_scores[node_id] = propagated_score * error_prob
            
            # 获取根因节点（使用最终分数，排除错误汇点）
            root_cause_id = max(
                {nid: score for nid, score in final_scores.items() 
                 if trace.error_sink_node_id != nid}.items(),
                key=lambda x: x[1]
            )[0] if final_scores else ""
            
            # 构建诊断结果（使用最终分数）
            blame_distribution = []
            for node in trace.nodes:
                final_score = final_scores.get(node.node_id, 0.0)
                propagated_score = blame_scores.get(node.node_id, 0.0)
                error_prob = error_probs.get(node.node_id, 0.0)
                
                diagnosis = self.generate_diagnosis(node.node_id, final_score, root_cause_id)
                
                blame_distribution.append(BlameDistribution(
                    node_id=node.node_id,
                    agent_role=node.agent_role,
                    node_type=node.node_type,
                    blame_score=final_score,  # 使用最终分数
                    diagnosis=diagnosis
                ))
            
            # 按责任分数降序排列
            blame_distribution.sort(key=lambda x: x.blame_score, reverse=True)
            
            # 获取根因节点的角色
            root_cause_node = graph_builder.get_node(root_cause_id)
            root_cause_agent_role = root_cause_node.agent_role if root_cause_node else ""
            
            return OutputResult(
                trace_id=trace.trace_id,
                status="success",
                diagnostic_results=DiagnosticResults(
                    root_cause_node_id=root_cause_id,
                    root_cause_agent_role=root_cause_agent_role,
                    blame_distribution=blame_distribution
                ),
                metrics=self.metrics
            )
        
        except Exception as e:
            print(f"处理错误: {e}")
            import traceback
            traceback.print_exc()
            return OutputResult(
                trace_id=input_data.get("trace_id", "unknown"),
                status="error",
                diagnostic_results=DiagnosticResults(
                    root_cause_node_id="",
                    root_cause_agent_role="",
                    blame_distribution=[]
                ),
                metrics=self.metrics
            )


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python main.py <input_json_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # 读取输入JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    # 创建系统并处理
    system = ASCMASSystem()
    result = system.process(input_data)
    
    # 输出结果
    output_dict = result.to_dict()
    print(json.dumps(output_dict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
