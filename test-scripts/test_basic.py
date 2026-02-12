"""
基础测试：测试图构建和传播算法（不需要API）
"""
import json
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import InputTrace
from graph_builder import GraphBuilder
from propagation import DampedBackwardPropagation


def test_graph_building():
    """测试图构建"""
    print("=" * 60)
    print("测试1：图构建")
    print("=" * 60)
    
    # 读取测试数据
    test_file = Path(__file__).parent.parent / "data_test" / "test_001.json"
    with open(test_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    trace = InputTrace.from_dict(input_data)
    graph_builder = GraphBuilder(trace)
    
    # 检查是否有环
    has_cycle = graph_builder.has_cycle()
    print(f"图是否有环: {has_cycle}")
    assert not has_cycle, "图不应该有环"
    
    # 获取逆拓扑排序
    reverse_order = graph_builder.get_reverse_topological_order()
    print(f"逆拓扑排序: {reverse_order}")
    assert len(reverse_order) > 0, "应该能够进行拓扑排序"
    
    # 检查error_sink_node是否在最后
    error_sink = trace.error_sink_node_id
    assert reverse_order[0] == error_sink, "错误汇点应该在逆拓扑排序的第一位"
    
    print("✓ 图构建测试通过\n")


def test_propagation():
    """测试传播算法（使用mock权重）"""
    print("=" * 60)
    print("测试2：阻尼反向传播算法")
    print("=" * 60)
    
    # 读取测试数据
    test_file = Path(__file__).parent.parent / "data_test" / "test_001.json"
    with open(test_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    trace = InputTrace.from_dict(input_data)
    graph_builder = GraphBuilder(trace)
    
    # 创建mock边权重（所有边权重相等）
    edge_weights = {}
    for child_node in trace.nodes:
        for parent_id in child_node.parent_ids:
            edge_weights[(parent_id, child_node.node_id)] = 0.5
    
    # 执行传播
    propagation = DampedBackwardPropagation(graph_builder, edge_weights)
    blame_scores = propagation.propagate(trace.error_sink_node_id)
    
    # 检查结果
    print("责任分数分布:")
    for node_id, score in sorted(blame_scores.items(), key=lambda x: x[1], reverse=True):
        node = graph_builder.get_node(node_id)
        print(f"  {node_id} ({node.agent_role}): {score:.3f}")
    
    # 验证：所有分数应该非负
    assert all(score >= 0 for score in blame_scores.values()), "所有分数应该非负"
    
    # 验证：错误汇点应该有分数
    assert blame_scores[trace.error_sink_node_id] > 0, "错误汇点应该有分数"
    
    # 获取根因
    root_cause_id, root_cause_score = propagation.get_root_cause()
    print(f"\n根因节点: {root_cause_id} (分数: {root_cause_score:.3f})")
    assert root_cause_id, "应该能找到根因节点"
    
    print("✓ 传播算法测试通过\n")


def main():
    """运行所有基础测试"""
    print("\n基础功能测试\n")
    
    try:
        test_graph_building()
        test_propagation()
        
        print("=" * 60)
        print("✓ 所有基础测试通过！")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
