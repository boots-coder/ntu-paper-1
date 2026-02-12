"""
测试脚本：运行所有测试数据集并保存结果
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import ASCMASSystem


def run_tests():
    """运行所有测试"""
    # 获取测试数据目录
    test_data_dir = Path(__file__).parent.parent / "data_test"
    output_base_dir = Path(__file__).parent.parent / "output"
    
    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有测试JSON文件
    test_files = sorted(test_data_dir.glob("*.json"))
    
    if not test_files:
        print("未找到测试文件！")
        return
    
    print(f"找到 {len(test_files)} 个测试文件")
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    
    # 创建系统实例
    system = ASCMASSystem()
    
    results = []
    success_count = 0
    error_count = 0
    
    # 运行每个测试
    for test_file in test_files:
        print(f"\n处理: {test_file.name}")
        print("-" * 60)
        
        try:
            # 读取输入
            with open(test_file, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            
            # 处理
            result = system.process(input_data)
            
            # 保存结果
            output_file = output_dir / f"{test_file.stem}_result.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
            
            # 记录结果
            results.append({
                "test_file": test_file.name,
                "trace_id": result.trace_id,
                "status": result.status,
                "root_cause": result.diagnostic_results.root_cause_node_id,
                "output_file": str(output_file)
            })
            
            if result.status == "success":
                success_count += 1
                print(f"✓ 成功 - 根因节点: {result.diagnostic_results.root_cause_node_id}")
                print(f"  责任分数分布:")
                for bd in result.diagnostic_results.blame_distribution[:3]:  # 只显示前3个
                    print(f"    {bd.node_id}: {bd.blame_score:.3f} - {bd.diagnosis}")
            else:
                error_count += 1
                print(f"✗ 失败 - 状态: {result.status}")
        
        except Exception as e:
            error_count += 1
            print(f"✗ 异常: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "test_file": test_file.name,
                "status": "error",
                "error": str(e)
            })
    
    # 保存汇总结果
    summary = {
        "timestamp": timestamp,
        "total_tests": len(test_files),
        "success_count": success_count,
        "error_count": error_count,
        "success_rate": success_count / len(test_files) if test_files else 0,
        "results": results
    }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    print(f"总测试数: {len(test_files)}")
    print(f"成功: {success_count}")
    print(f"失败: {error_count}")
    print(f"成功率: {summary['success_rate']:.2%}")
    print(f"\n结果保存在: {output_dir}")
    print(f"汇总文件: {summary_file}")


if __name__ == "__main__":
    run_tests()
