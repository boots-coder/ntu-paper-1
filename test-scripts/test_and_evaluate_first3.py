"""
测试前三个测试用例并评估结果
"""
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import ASCMASSystem


def test_first_three():
    """测试前三个测试文件并评估"""
    test_data_dir = Path(__file__).parent.parent / "data_test"
    output_base_dir = Path(__file__).parent.parent / "output"
    
    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 只测试前3个
    test_files = sorted(test_data_dir.glob("*.json"))[:3]
    
    print(f"测试前3个测试文件")
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    
    system = ASCMASSystem()
    results = []
    
    # Ground truth
    ground_truth = {
        "math_task_001": "node_004",
        "logic_task_002": "node_002",
        "calculation_task_003": "node_003",
    }
    
    correct_count = 0
    
    for test_file in test_files:
        print(f"\n处理: {test_file.name}")
        print("-" * 60)
        
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            
            result = system.process(input_data)
            
            # 保存结果
            output_file = output_dir / f"{test_file.stem}_result.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
            
            trace_id = result.trace_id
            predicted = result.diagnostic_results.root_cause_node_id
            expected = ground_truth.get(trace_id, "")
            
            is_correct = (predicted == expected)
            if is_correct:
                correct_count += 1
            
            status = "✓" if is_correct else "✗"
            print(f"{status} 预测: {predicted}, 期望: {expected}")
            
            # 显示前3个节点的分数
            print("  责任分数分布:")
            for bd in result.diagnostic_results.blame_distribution[:3]:
                print(f"    {bd.node_id}: {bd.blame_score:.3f}")
            
            results.append({
                "test_file": test_file.name,
                "trace_id": trace_id,
                "predicted": predicted,
                "expected": expected,
                "correct": is_correct
            })
        
        except Exception as e:
            print(f"✗ 异常: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "test_file": test_file.name,
                "status": "error",
                "error": str(e)
            })
    
    # 保存评估结果
    evaluation = {
        "timestamp": timestamp,
        "total_tests": len(test_files),
        "correct_predictions": correct_count,
        "accuracy": correct_count / len(test_files) if test_files else 0.0,
        "results": results
    }
    
    eval_file = output_dir / "evaluation.json"
    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2)
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"总测试数: {len(test_files)}")
    print(f"正确预测: {correct_count}")
    print(f"准确率: {evaluation['accuracy']:.2%}")
    print(f"\n结果保存在: {output_dir}")
    print(f"评估文件: {eval_file}")


if __name__ == "__main__":
    test_first_three()
