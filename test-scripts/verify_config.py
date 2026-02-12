"""
验证配置文件：检查API key是否配置正确
"""
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OPENAI_API_KEY, OPENAI_MODEL

# 尝试导入新版本的OpenAI客户端
try:
    from openai import OpenAI
    OPENAI_NEW_API = True
except ImportError:
    try:
        import openai
        OPENAI_NEW_API = False
    except ImportError:
        OPENAI_NEW_API = None


def verify_openai_key():
    """验证OpenAI API key"""
    print("=" * 60)
    print("验证OpenAI API配置")
    print("=" * 60)
    
    if not OPENAI_API_KEY:
        print("✗ 错误：OPENAI_API_KEY未设置")
        print("  请在config.py中设置OPENAI_API_KEY，或设置环境变量OPENAI_API_KEY")
        return False
    
    print(f"✓ API Key已设置（长度: {len(OPENAI_API_KEY)}）")
    print(f"✓ 使用模型: {OPENAI_MODEL}")
    
    if OPENAI_NEW_API is None:
        print("✗ 错误：未安装openai包")
        return False
    
    # 尝试调用API
    try:
        if OPENAI_NEW_API:
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "user", "content": "Hello"}
                ],
                max_tokens=10
            )
        else:
            openai.api_key = OPENAI_API_KEY
            response = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "user", "content": "Hello"}
                ],
                max_tokens=10
            )
        print("✓ API调用成功")
        return True
    except Exception as e:
        print(f"✗ API调用失败: {e}")
        print("  请检查API key是否正确，以及是否有足够的配额")
        return False


def verify_dependencies():
    """验证依赖包"""
    print("\n" + "=" * 60)
    print("验证依赖包")
    print("=" * 60)
    
    dependencies = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "sentence_transformers": "Sentence Transformers",
        "networkx": "NetworkX",
        "numpy": "NumPy",
        "openai": "OpenAI"
    }
    
    all_ok = True
    for module_name, display_name in dependencies.items():
        try:
            __import__(module_name)
            print(f"✓ {display_name}")
        except ImportError:
            print(f"✗ {display_name} 未安装")
            all_ok = False
    
    return all_ok


if __name__ == "__main__":
    print("\n配置验证工具\n")
    
    deps_ok = verify_dependencies()
    api_ok = verify_openai_key()
    
    print("\n" + "=" * 60)
    if deps_ok and api_ok:
        print("✓ 所有配置验证通过！")
        sys.exit(0)
    else:
        print("✗ 配置验证失败，请修复上述问题")
        sys.exit(1)
