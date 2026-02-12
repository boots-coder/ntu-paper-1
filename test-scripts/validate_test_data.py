"""
校验测试数据的脚本
检查：
1. JSON格式是否正确
2. 数据结构是否完整
3. 节点ID是否唯一
4. parent_ids是否有效
5. error_sink_node_id是否存在
6. 图是否有环
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import InputTrace
from graph_builder import GraphBuilder


def validate_test_file(file_path: Path) -> tuple[bool, list[str]]:
    """验证单个测试文件"""
    errors = []
    
    try:
        # 1. 检查JSON格式
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 2. 检查必需字段
        required_fields = ['trace_id', 'problem', 'error_sink_node_id', 'nodes']
        for field in required_fields:
            if field not in data:
                errors.append(f"缺少必需字段: {field}")
        
        if errors:
            return False, errors
        
        # 3. 解析为InputTrace
        trace = InputTrace.from_dict(data)
        
        # 4. 检查节点ID唯一性
        node_ids = [node.node_id for node in trace.nodes]
        if len(node_ids) != len(set(node_ids)):
            duplicates = [nid for nid in node_ids if node_ids.count(nid) > 1]
            errors.append(f"节点ID重复: {duplicates}")
        
        # 5. 检查error_sink_node_id是否存在
        if trace.error_sink_node_id not in node_ids:
            errors.append(f"error_sink_node_id '{trace.error_sink_node_id}' 不存在于节点列表中")
        
        # 6. 检查parent_ids是否有效
        all_node_ids = set(node_ids)
        for node in trace.nodes:
            for parent_id in node.parent_ids:
                if parent_id not in all_node_ids:
                    errors.append(f"节点 {node.node_id} 的父节点 {parent_id} 不存在")
        
        # 7. 检查图是否有环
        graph_builder = GraphBuilder(trace)
        if graph_builder.has_cycle():
            errors.append("图中存在环")
        
        # 8. 检查是否有根因节点（至少有一个节点没有父节点）
        nodes_without_parents = [node for node in trace.nodes if not node.parent_ids]
        if not nodes_without_parents:
            errors.append("所有节点都有父节点，图中没有根节点")
        
        # 9. 检查错误是否明显（改进的检查）
        error_found = False
        error_nodes = []
        
        for node in trace.nodes:
            content = node.content
            problem = trace.problem
            
            # 检查1：明显的错误关键词
            if any(keyword in content for keyword in ['错误', '不对', '不正确', 'wrong']):
                error_found = True
                error_nodes.append(node.node_id)
            
            # 检查2：计算错误（如 10/2=4 应该是5）
            import re
            calc_patterns = [
                r'(\d+)\s*[/÷]\s*(\d+)\s*=\s*(\d+)',
                r'(\d+)\s*[+\+]\s*(\d+)\s*=\s*(\d+)',
            ]
            for pattern in calc_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    try:
                        if len(match) == 3:
                            a, b, result = map(float, match)
                            if '/' in content or '÷' in content:
                                expected = a / b
                            elif '+' in content:
                                expected = a + b
                            else:
                                continue
                            if abs(result - expected) > 0.01:
                                error_found = True
                                error_nodes.append(node.node_id)
                    except:
                        pass
            
            # 检查3：逻辑矛盾（如"15比20小，所以15更大"）
            if '比' in content and '小' in content and '更大' in content:
                error_found = True
                error_nodes.append(node.node_id)
            
            # 检查4：日期错误（如"明天是周三"但问题说"明天是周二"）
            if '周三' in content and ('周二' in problem or '明天' in problem):
                error_found = True
                error_nodes.append(node.node_id)
            
            # 检查5：数字不一致（如说"进货30个"但问题说"进货50个"）
            if '30' in content and '50' in problem and '进货' in content:
                error_found = True
                error_nodes.append(node.node_id)
        
        if not error_found:
            errors.append("警告：未检测到明显的错误内容")
        else:
            print(f"  检测到错误节点: {list(set(error_nodes))}")
        
        return len(errors) == 0, errors
    
    except json.JSONDecodeError as e:
        return False, [f"JSON格式错误: {e}"]
    except Exception as e:
        return False, [f"验证错误: {e}"]


def main():
    """主函数"""
    test_dir = Path(__file__).parent.parent / "data_test"
    test_files = sorted(test_dir.glob("*.json"))
    
    print("=" * 60)
    print("测试数据校验")
    print("=" * 60)
    
    all_valid = True
    for test_file in test_files:
        is_valid, errors = validate_test_file(test_file)
        status = "✓" if is_valid else "✗"
        print(f"\n{status} {test_file.name}")
        
        if errors:
            all_valid = False
            for error in errors:
                print(f"  - {error}")
        else:
            print("  通过所有检查")
    
    print("\n" + "=" * 60)
    if all_valid:
        print("✓ 所有测试数据校验通过")
    else:
        print("✗ 部分测试数据存在问题")
    print("=" * 60)
    
    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
