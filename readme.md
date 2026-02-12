
# 架构与实现白皮书：基于自适应语义-因果融合图的 MAS 错误溯源系统 (ASC-MAS)

## 1. 系统工程总览 (System Overview)

本系统 (ASC-MAS) 接收多智能体系统 (MAS) 运行失败后的交互日志，通过构建**混合依赖有向无环图 (Hybrid Dependency DAG)**，并在图上执行**马尔可夫能量衰减传播**，精准定位导致系统崩溃的“根因节点 (Root Cause)”及“失职审查节点 (Critic Failure)”。

**核心技术栈定位：**

* **基础流 (语义追踪):** 基于轻量级模型 (如 DistilBERT) 的 Attention 矩阵，计算节点间的纯文本注意力对齐度。
* **兜底流 (因果推断):** 基于大模型 (LLM) 的蒙特卡洛反事实采样，计算节点间真实的逻辑因果效应。
* **路由层 (自适应门控):** 基于节点特征的轻量级 MLP，动态计算上述两股数据流的融合权重。

---

## 2. 数据契约与接口规范 (Data Contract & I/O)

研发团队需首先定义系统的数据输入输出 (DTO) 格式。这是整个数据流转的起点和终点。

### 2.1 输入数据规范 (Input JSON)

输入为单次失败任务的完整轨迹 (Trace)。`parent_ids` 字段是系统进行图拓扑排序的唯一依据。

**示例场景：** 算错进货量 + 审查员盲目通过。

```json
{
  "trace_id": "math_task_088",
  "problem": "仓库有100个苹果，卖出20个，又进货50个。现在有几个？",
  "error_sink_node_id": "node_006", 
  "nodes": [
    {
      "node_id": "node_001",
      "agent_role": "Proposer",
      "node_type": "Thought",
      "content": "第一步，计算卖出后的剩余数量：100 - 20。",
      "parent_ids": [] 
    },
    {
      "node_id": "node_002",
      "agent_role": "Tool",
      "node_type": "Observation",
      "content": "80",
      "parent_ids": ["node_001"]
    },
    {
      "node_id": "node_004",
      "agent_role": "Proposer",
      "node_type": "Thought",
      "content": "第二步，加上新进货的数量。仓库又进货了 30 个苹果。所以 80 + 30 = 110。",
      "parent_ids": ["node_002"] 
    },
    {
      "node_id": "node_005",
      "agent_role": "Critic",
      "node_type": "Review",
      "content": "审查完毕。以上所有计算步骤逻辑清晰，完全正确。",
      "parent_ids": ["node_001", "node_002", "node_004"] 
    },
    {
      "node_id": "node_006",
      "agent_role": "Summarizer",
      "node_type": "Output",
      "content": "因此，仓库现在共有 110 个苹果。",
      "parent_ids": ["node_004", "node_005"] 
    }
  ]
}

```

### 2.2 输出数据规范 (Output JSON)

系统处理完毕后，输出每个节点的**量化责任分数 (`blame_score`)** 以及基于分数的**诊断结论 (`diagnosis`)**。

```json
{
  "trace_id": "math_task_088",
  "status": "success",
  "diagnostic_results": {
    "root_cause_node_id": "node_004",
    "root_cause_agent_role": "Proposer",
    "blame_distribution": [
      {
        "node_id": "node_004",
        "agent_role": "Proposer",
        "node_type": "Thought",
        "blame_score": 0.615,
        "diagnosis": "首要责任节点 (Root Cause)。因果推断判定其对错误输出有强干预效应。"
      },
      {
        "node_id": "node_005",
        "agent_role": "Critic",
        "node_type": "Review",
        "blame_score": 0.285,
        "diagnosis": "次要/连带责任节点。存在审查失职，未能阻断错误链路。"
      },
      {
        "node_id": "node_002",
        "agent_role": "Tool",
        "node_type": "Observation",
        "blame_score": 0.100,
        "diagnosis": "背景传导节点 (阻尼衰减截留部分分数)。"
      },
      {
        "node_id": "node_001",
        "agent_role": "Proposer",
        "node_type": "Thought",
        "blame_score": 0.000,
        "diagnosis": "无责任。"
      }
    ]
  },
  "metrics": {
    "semantic_engine_invocations": 5,
    "causal_engine_invocations": 5,
    "monte_carlo_samples_generated": 25
  }
}

```

---

## 3. 核心算法工作引擎 (Core Engines Implementation)

开发团队需构建以下三大引擎来计算图中任意有向边  的权重 。

### 3.1 引擎 A：语义注意力追踪器 (Semantic Attention Tracer)

**工程目标：** 快速计算文本相似度与注意力转移，输出 。
**实现步骤：**

1. **加载基座：** 部署本地 `distilbert-base-uncased` 模型。
2. **输入构造：** 将父节点内容  和子节点内容  拼接为 `[CLS] text_p [SEP] text_c [SEP]` 输入模型。
3. **提取特征：** 提取 Transformer 最后一层的自注意力矩阵 (Attention Matrix)。
4. **计算得分：** * 在矩阵中，截取从  的 Tokens 指向  的 Tokens 的注意力子矩阵。
* 沿子节点 Token 维度取平均值（Mean Pooling），得到标量 。这代表子节点生成时有多大程度“盯”着父节点。



### 3.2 引擎 B：蒙特卡洛因果推断器 (Monte Carlo Causal Auditor)

**工程目标：** 通过扰动变量，利用 LLM 计算真实的因果依赖度，输出 。

**实现步骤：**

1. **自动扰动 (Perturbation)：** 对父节点  生成  个扰动版本 。
* *扰动策略：* 若正则提取出数字，将其  或替换为随机数；若无数字，随机 Mask 掉 15% 的名词/动词。


2. **反事实模拟 (Simulation)：** 并发请求 LLM（如 GPT-4o-mini）。
* *Prompt:* `背景上下文是 [...], 现在给定前提 {扰动后的父节点}, 请扮演原智能体角色，重新生成下一步的响应。`
* *获取输出:* 。


3. **计算散度 (Divergence)：** 计算真实的子节点  与  个生成的假子节点  的语义距离 (可使用 Sentence-BERT 提取 Embedding 计算余弦距离 )。
4. **计算因果得分：** 。
* *物理意义：* 如果父节点一变，子节点跟着大变（距离大），说明强因果，得分高。



### 3.3 引擎 C：自适应融合路由器 (Adaptive Fusion Router)

**工程目标：** 计算门控参数 ，融合两股数据流。
**实现步骤：**

1. **特征工程 (Feature Extraction)：** 针对当前边，提取 3 个特征构成的向量 ：
* : 父子节点是否包含数值 (0或1)。
* : 子节点的类型 One-hot (如 Critic=1, Thought=0)。
* :  (引擎 A 算出的先验语义得分)。


2. **路由计算：** 设计一个单层 MLP（可通过人工标注少量数据预训练，或初始化为均权）：

3. **融合输出边权重 ：**



---

## 4. 溯源算法：阻尼反向传播 (Damped Backward Propagation)

一旦图  及所有边权重  构建完毕，系统进入马尔可夫能量回传阶段。

### 4.1 数据初始化

* 令所有节点责任分数 。
* 根据输入 JSON 的 `error_sink_node_id`，注入初始“错误能量”：。
* 获取图  的**逆拓扑排序序列** (保证总是从结果向原因遍历)。

### 4.2 传播主循环 (核心公式)

引入**阻尼系数 ** (Damping Factor，防止长上下文中的梯度爆炸或无限追溯，强迫责任停留在最直接的出错节点上)。

遍历逆拓扑序列中的每个节点 。若 ，则将其责任分摊给其所有父节点 。

1. **计算当前节点需向上分配的总能量：**



*(注：剩下的  能量截留在  自身，作为它的历史责任。)*
2. **计算局部归一化权重分母：**


3. **按权重比例回传给父节点 ：**
* 如果 ：


* 如果  (逻辑彻底断链的极端兜底)：





### 4.3 终态输出

* 循环结束后，按  对所有节点进行降序排列。
* 取  值最高者作为 Root Cause 并填充输出 JSON。

---

## 5. 工程实施路线图 (Implementation Roadmap)

开发团队需按照以下顺序分阶段交付：

* **Phase 1: 基础设施搭建与纯图算法跑通 (Week 1)**
* 定义 Pydantic / dataclass 模型以解析 Input JSON。
* 引入 `networkx` 库完成节点的注入与拓扑排序。
* **Mock 测试：** 手动硬编码一个权重矩阵 ，跑通 【第 4 节】的阻尼反向传播算法，验证输出的 Blame Score 能够正确加和与衰减。


* **Phase 2: 语义流与因果流引擎介入 (Week 2-3)**
* 封装 HuggingFace 的 DistilBERT，实现注意力分数的提取 (引擎 A)。
* 封装 OpenAI/VLLM 客户端，实现蒙特卡洛扰动生成、并发请求和文本距离计算 (引擎 B)。


* **Phase 3: 路由融合与全链路联调 (Week 4)**
* 实现特征提取器和 MLP 路由。
* 输入真实的测试集 (如我们在输入示例中给定的带注入错误的 trace)，观察系统能否自动算出边权重，并最终将 `node_004` (Root Cause) 的 Score 顶到第一名。



此文档现已打通了学术级因果推断理论与工业级微服务架构的桥梁。研发人员只需按照文中的公式和数据结构“按图索骥”，即可实现这套 SOTA (State-of-the-Art) 级别的多智能体纠错系统。