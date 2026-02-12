"""
评估脚本：评估测试结果的准确率
需要提供ground truth（正确答案）来评估根因节点识别是否准确
"""
import json
import sys
from pathlib import Path
from typing import Dict, List


def load_ground_truth() -> Dict[str, str]:
    """
    加载ground truth（正确答案）
    返回字典：{trace_id: expected_root_cause_node_id}
    """
    # 这里可以根据实际情况定义ground truth
    # 暂时使用基于测试数据的预期结果
    ground_truth = {
        "math_task_001": "node_004",  # 错误：说进货30个，实际应该是50个
        "logic_task_002": "node_002",  # 错误：说明天是周三，实际应该是周二
        "calculation_task_003": "node_003",  # 错误：10/2=4，实际应该是5
        "comparison_task_006": "node_002",  # 错误：15比20小，所以15更大（逻辑错误）
    }
    return ground_truth


def evaluate_results(output_dir: Path, ground_truth: Dict[str, str]) -> Dict:
    """
    评估结果
    
    Args:
        output_dir: 输出目录路径
        ground_truth: ground truth字典
    
    Returns:
        评估结果字典
    """
    # 读取summary.json
    summary_file = output_dir / "summary.json"
    if not summary_file.exists():
        print(f"错误：找不到summary文件 {summary_file}")
        return {}
    
    with open(summary_file, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    # 评估每个结果
    correct_predictions = 0
    total_predictions = 0
    detailed_results = []
    
    for result in summary.get("results", []):
        if result.get("status") != "success":
            continue
        
        trace_id = result.get("trace_id", "")
        predicted_root_cause = result.get("root_cause", "")
        expected_root_cause = ground_truth.get(trace_id, "")
        
        if expected_root_cause:
            total_predictions += 1
            is_correct = (predicted_root_cause == expected_root_cause)
            
            if is_correct:
                correct_predictions += 1
            
            detailed_results.append({
                "trace_id": trace_id,
                "test_file": result.get("test_file", ""),
                "predicted": predicted_root_cause,
                "expected": expected_root_cause,
                "correct": is_correct
            })
    
    # 计算准确率
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    evaluation_result = {
        "timestamp": summary.get("timestamp", ""),
        "total_predictions": total_predictions,
        "correct_predictions": correct_predictions,
        "accuracy": accuracy,
        "detailed_results": detailed_results
    }
    
    return evaluation_result


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python evaluate.py <output_directory>")
        print("示例: python evaluate.py ../output/20240212_143000")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    
    if not output_dir.exists():
        print(f"错误：目录不存在 {output_dir}")
        sys.exit(1)
    
    # 加载ground truth
    ground_truth = load_ground_truth()
    
    # 评估结果
    evaluation = evaluate_results(output_dir, ground_truth)
    
    if not evaluation:
        print("评估失败")
        sys.exit(1)
    
    # 保存评估结果
    eval_file = output_dir / "evaluation.json"
    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2)
    
    # 打印评估结果
    print("=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"总预测数: {evaluation['total_predictions']}")
    print(f"正确预测: {evaluation['correct_predictions']}")
    print(f"准确率: {evaluation['accuracy']:.2%}")
    print("\n详细结果:")
    for detail in evaluation['detailed_results']:
        status = "✓" if detail['correct'] else "✗"
        print(f"  {status} {detail['trace_id']}: "
              f"预测={detail['predicted']}, "
              f"期望={detail['expected']}")
    
    print(f"\n评估结果保存在: {eval_file}")


if __name__ == "__main__":
    main()
