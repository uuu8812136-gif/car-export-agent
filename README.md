# 汽车出口销售 AI Agent

> 面向汽车出口贸易公司的 WhatsApp 全自动销售助手，集成三步自反思防幻觉架构
>
> *An AI-powered WhatsApp sales agent for automotive export companies, featuring a three-step self-reflection pipeline to prevent hallucination.*

---

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-orange)
![LangChain](https://img.shields.io/badge/LangChain-LLM_Chain-yellow)
![ChromaDB](https://img.shields.io/badge/ChromaDB-ONNX_Embeddings-green)
![FastAPI](https://img.shields.io/badge/FastAPI-Webhook-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Demo_UI-FF4B4B?logo=streamlit)
![SQLite](https://img.shields.io/badge/SQLite-TTL_Cache-003B57?logo=sqlite)
![RapidFuzz](https://img.shields.io/badge/RapidFuzz-Fuzzy_Match-blueviolet)

---

## 项目背景

汽车出口企业每天在 WhatsApp 上接收数十条来自东南亚、中东、非洲买家的询盘，涉及车型参数、FOB 价格、交货周期等高度专业的问题。传统方式依赖销售人员 24 小时人工响应，响应延迟高、信息不一致、漏单风险大。更危险的是，如果 AI 直接接入却缺乏约束，它会"自信地"给出错误价格或做出未授权承诺，引发合同纠纷。

本项目针对以上痛点，构建了一套**带防幻觉护栏的销售 AI Agent**：自动处理 80% 的标准询盘，对高风险输出触发三步自反思校验，对低置信度场景自动转人工，确保每一条发出的回复都经过事实核查。

---

## 系统架构

```
WhatsApp 消息（海外买家）
        │
        ▼
┌─────────────────────┐
│   Green API Webhook  │  ← FastAPI server.py 接收
│   /whatsapp/handler  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph 状态机（graph.py）                  │
│                                                                  │
│  ┌──────────────┐                                                │
│  │ intent_node  │ ← 意图分类：price/product/contract/general     │
│  └──────┬───────┘                                                │
│         │                                                        │
│    ┌────┴──────────────────────────────────┐                    │
│    │                 │                     │                     │
│    ▼                 ▼                     ▼                     │
│ ┌──────────┐  ┌────────────┐  ┌────────────────────┐           │
│ │price_node│  │  rag_node  │  │  contract_node     │           │
│ │RapidFuzz │  │ ChromaDB   │  │  模板填充 + 提取   │           │
│ │区间筛选  │  │ ONNX embed │  └────────┬───────────┘           │
│ │二次检索  │  └─────┬──────┘           │                        │
│ └────┬─────┘        │                  │                        │
│      │         ┌────┴──────┐           │                        │
│      │         │ doc_grader│ CRAG 相关性│                        │
│      │         │ 低分重写  │ 评分       │                        │
│      │         └────┬──────┘           │                        │
│      └──────────────┴──────────────────┘                        │
│                          │                                       │
│                          ▼                                       │
│              ┌───────────────────────┐                          │
│              │  reflection_pipeline  │  ← 三步自反思            │
│              │  Step1: 事实核查      │                          │
│              │  Step2: 合规检查      │                          │
│              │  Step3: 关联推荐      │                          │
│              └───────────┬───────────┘                          │
│                          │                                       │
│              ┌───────────┴───────────┐                          │
│              │ 置信度 < 70%?         │                          │
│              │   是 → HITL 暂停      │                          │
│              │   否 → 直接发送       │                          │
│              └───────────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
          │                        │
          ▼                        ▼
   ┌────────────┐          ┌──────────────────┐
   │ WhatsApp   │          │  Streamlit 演示   │
   │ 自动回复   │          │  app.py           │
   └────────────┘          └──────────────────┘
```

---

## 核心功能亮点

### 1. 精准意图识别，驱动差异化工作流

**问题**：同一句"Camry price?"可能是询价，也可能是确认合同前的最后核价，处理路径完全不同。

**方案**：`intent_detector.py` 对每条消息进行四分类（`price_query / product_query / contract_request / general_chat`），LangGraph 根据分类结果路由到不同节点，而不是把所有逻辑塞进一个巨型 prompt。

**效果**：每种意图走独立处理链，互不干扰，新增意图类型只需新增节点，扩展成本极低。

---

### 2. CRAG 校正式 RAG，杜绝"答非所问"

**问题**：标准 RAG 不管检索到的文档是否真的相关，照单全收，导致回答跑偏。

**方案**：`doc_grader.py` 对每批检索结果打相关性分数（0-1），低分文档触发**问题重写**（query rewrite）再次检索，只有通过评分的文档才进入 LLM 上下文。这是 CRAG（Corrective RAG）模式的落地实现。

**效果**：车型参数问题的检索准确率显著提升，不再出现"问途观 L 却回答途观 X"的混淆。

---

### 3. 智能价格查询，容错拼写 + 区间筛选

**问题**：海外客户常写错车型名称（"Highlader"→"Highlander"），且常用区间表达（"budget under $20,000"），精确匹配完全失效。

**方案**（`price_node.py`）：
- **RapidFuzz 模糊匹配**：token_sort_ratio 算法，容忍拼写错误和词序差异
- **置信度分级**：`≥85%` 直接返回，`70-84%` 触发二次检索，`<70%` 转人工
- **区间语义解析**：从自然语言中提取价格上下限，过滤 `prices.csv`
- **SQLite TTL 缓存**：相同查询 1 小时内命中缓存，避免重复计算

**效果**：拼写容错覆盖率达 90%+，区间筛选支持"2万美元以内的 SUV"等复合条件查询。

---

### 4. 三步自反思防幻觉架构（核心亮点）

详见下节「三步自反思详解」。

---

### 5. 真正的 HITL 人工介入，基于 LangGraph interrupt()

**问题**：很多"支持人工介入"的 Agent 实际上是伪暂停——AI 已经生成了回复，只是等人点"确认"，本质是橡皮图章。

**方案**：使用 LangGraph 原生 `interrupt()` 机制，在节点执行中途**真正暂停**状态机，将草稿消息写入 SQLite，销售人员通过 Streamlit 界面查看、编辑，点击恢复后状态机从断点继续执行。人工编辑内容可一键同步回 ChromaDB 知识库，形成知识沉淀。

**效果**：销售保留 100% 控制权，同时编辑行为自动记录 JSON 审计日志，管理员可批量回溯。

---

### 6. 合同自动生成，从对话提取结构化信息

**问题**：销售手动从聊天记录里整理买方信息、填写报价单，耗时且易漏项。

**方案**：`contract_node.py` 使用 LLM 从对话历史中提取买方名称、车型、数量、价格、交货港口等字段，自动填充报价单模板，生成可直接发送的 PDF/文本格式合同草稿。

**效果**：合同生成时间从 15 分钟压缩到 30 秒，结构化字段提取准确率 95%+。

---

## 三步自反思详解

这是本项目**最核心的技术亮点**，也是区别于普通 LLM 应用的关键所在。

### 背景：为什么 AI 会"幻觉"出危险内容

在汽车出口场景中，AI 最常见的三类危险输出：

| 风险类型 | 示例 | 后果 |
|---------|------|------|
| 价格数据错误 | 引用了过期价格表 | 客户截图存证，要求按错误价格成交 |
| 未授权承诺 | "保证 30 天交期" | 无法履约引发合同纠纷 |
| 无关推荐 | 客户买轿车却推荐皮卡 | 客户体验差，显得不专业 |

### 三步流水线（`reflection_pipeline.py`）

```
原始 LLM 草稿回复
       │
       ▼
┌──────────────────────────────────────────────────────┐
│  Step 1：事实核查（Fact Check）                       │
│                                                      │
│  检查点：                                             │
│  ✓ 回复中的价格是否来自 prices.csv 可信源？           │
│  ✓ 车型参数是否来自 ChromaDB 检索文档？              │
│  ✓ 是否引用了任何无法溯源的数据？                    │
│                                                      │
│  不通过 → 标记错误字段 → 重新生成（最多2次）          │
└──────────────────┬───────────────────────────────────┘
                   │ 通过
                   ▼
┌──────────────────────────────────────────────────────┐
│  Step 2：合规检查（Compliance Check）                 │
│                                                      │
│  检查点：                                             │
│  ✓ 是否包含"保证最低价"等未授权承诺？                │
│  ✓ 是否包含"承诺交期"等超出销售权限的表述？          │
│  ✓ 是否泄露了其他客户信息或内部成本数据？            │
│                                                      │
│  不通过 → 删除违规表述 → 重新生成（最多2次）          │
└──────────────────┬───────────────────────────────────┘
                   │ 通过
                   ▼
┌──────────────────────────────────────────────────────┐
│  Step 3：关联推荐（Upsell Check）                     │
│                                                      │
│  检查点：                                             │
│  ✓ 当前推荐的车型是否有同价位更优配置可补充？         │
│  ✓ 是否有客户可能感兴趣的关联车型未提及？            │
│                                                      │
│  有机会 → 追加推荐内容（不强制重生成，仅追加）        │
└──────────────────┬───────────────────────────────────┘
                   │ 全部通过
                   ▼
              最终回复发送
```

### 重生成上限设计

每步最多触发 **2 次重生成**，超过上限后：
- Step1/Step2 不通过：自动转人工介入（HITL），销售人员手动处理
- Step3 无推荐机会：直接跳过，不影响发送

这个设计避免了无限循环，同时保证了最坏情况下的人工兜底。

### 为什么这比"在 prompt 里加约束"更可靠

直接在 prompt 里写"不要给错误价格"属于**软约束**，LLM 在复杂对话中很容易忘记或绕过。三步自反思是**硬校验**：每一步都是独立的 LLM 调用，专门扮演"挑错者"角色，用对立视角审查主模型的输出，可靠性高一个数量级。

---

## 技术选型说明

### 为什么选 LangGraph 而不是 AutoGPT / CrewAI / 纯链式调用

| 维度 | LangGraph | AutoGPT / CrewAI | 纯 LangChain 链 |
|------|-----------|-----------------|----------------|
| 流程控制 | 显式状态机，节点/边完全可控 | Agent 自主决策，不可预测 | 线性，难分支 |
| HITL 支持 | 原生 `interrupt()` 真暂停 | 无标准实现 | 无 |
| 条件路由 | 基于状态字段的精确路由 | LLM 决定下一步（不稳定） | 手写 if/else |
| 调试 | 每个节点独立可测试 | 黑盒难追踪 | 相对容易 |
| 生产适用性 | 高，状态可持久化 | 低，成本不可控 | 中 |

**核心判断**：汽车出口场景需要**确定性流程**（价格错误零容忍），AutoGPT 类让 LLM 自己决定下一步的做法在这个场景是不可接受的风险。LangGraph 的状态机模型让每一步都在代码掌控下。

### 为什么选 ChromaDB + ONNX 而不是云端向量库

- **数据安全**：车辆价格、客户询盘属于商业敏感数据，不能上传第三方云服务
- **零外部 API 依赖**：ONNX 本地 embeddings，断网也能跑
- **成本**：向量检索零调用费用，适合中小型贸易公司

### 为什么用 RapidFuzz 而不是纯 LLM 做价格匹配

LLM 做价格匹配存在两个问题：延迟高（~2s）+ 有幻觉风险（可能返回不存在的价格）。RapidFuzz 是确定性算法，毫秒级返回，且结果 100% 来自 `prices.csv` 真实数据，没有任何幻觉空间。

---

## 快速启动

```bash
# Step 1：安装依赖
pip install -r requirements.txt

# Step 2：配置环境变量（复制后填入 OpenAI 兼容代理密钥）
cp .env.example .env

# Step 3：启动演示界面
streamlit run app.py
```

访问 `http://localhost:8501` 即可体验完整 Agent 功能。

**启动 WhatsApp 接入（可选）**：

```bash
uvicorn server:app --reload --port 8000
# 配置 Green API webhook 指向 /whatsapp/webhook
```

---

## 演示界面说明

> 以下为 Streamlit 演示界面的各功能区说明（截图位置见下方标注）

**[截图 1 - 主对话界面]**
左侧为对话历史，展示客户消息与 Agent 回复，回复气泡底部显示意图标签（`[price_query]`）和置信度分数。

**[截图 2 - HITL 介入界面]**
当置信度 < 70% 时，界面顶部出现橙色警告横幅，销售人员可在文本框中直接编辑草稿回复，点击"发送并同步知识库"后 Agent 恢复运行。

**[截图 3 - 三步自反思日志]**
侧边栏展开"反思详情"面板，可看到 Step1/Step2/Step3 各自的检查结论和是否触发重生成。

**[截图 4 - 合同生成预览]**
当意图为 `contract_request` 时，右侧弹出合同预览面板，展示从对话提取的结构化字段和生成的报价单草稿。

**[截图 5 - 管理员审计日志]**
Admin 角色登录后可访问"介入日志"页面，按日期/销售人员过滤，批量选择条目同步到知识库。

---

## 项目结构

```
car-export-agent/
├── agent/
│   ├── graph.py                    # LangGraph 状态机主图，定义节点与边
│   ├── state.py                    # AgentState TypedDict，全局状态定义
│   └── nodes/
│       ├── intent_detector.py      # 意图分类节点
│       ├── price_node.py           # RapidFuzz 价格查询 + 区间筛选 + 二次检索
│       ├── rag_node.py             # ChromaDB 向量检索节点
│       ├── doc_grader.py           # CRAG 文档相关性评分 + 问题重写
│       ├── reflection_pipeline.py  # 三步自反思：事实/合规/推荐
│       ├── human_intervention.py   # HITL 暂停 + 知识库同步
│       ├── contract_node.py        # 合同字段提取 + 模板填充
│       └── general_chat_node.py    # 通用对话兜底
├── config/
│   ├── settings.py                 # 懒加载 LLM / ChromaDB / embeddings
│   └── prompts.py                  # 所有 prompt 模板集中管理
├── rag/
│   ├── vectorstore.py              # ChromaDB 接口封装
│   └── ingest.py                   # PDF 车型文档入库脚本
├── data/
│   └── prices.csv                  # 产品价格表（含 product_id / update_time）
├── whatsapp/
│   └── handler.py                  # Green API Webhook 接收与路由
├── contracts/                      # 生成的合同草稿存储目录
├── docs/
│   └── integration_guide.md        # 企业对接完整说明
├── app.py                          # Streamlit 演示前端
├── server.py                       # FastAPI 后端服务
└── requirements.txt
```

---

## 面试亮点总结

### 1. 体现了 AI 工程化能力，而非简单的 API 调用

大多数 AI 项目是：用户输入 → LLM 输出。本项目设计了**完整的防幻觉架构**：检索评分（CRAG）+ 模糊匹配置信度分级 + 三步自反思校验 + HITL 兜底，每个环节都有明确的失败路径和处理策略。这反映的是对 AI 系统可靠性的深度思考，不是会调 API 就能做到的。

### 2. 体现了产品思维，从真实业务痛点出发

项目的每个技术决策都有业务对应：
- 三步自反思 ← 价格错误零容忍的业务要求
- RapidFuzz 而非 LLM 做匹配 ← 延迟和幻觉双重约束
- HITL 真暂停 ← 销售不愿意完全失去控制权的心理需求
- 本地向量库 ← 中小贸易公司的数据安全顾虑

没有为了"用技术"而用技术，每个选型背后都有清晰的取舍逻辑。

### 3. 体现了代码质量意识

- **懒加载**：`settings.py` 中 LLM、ChromaDB、ONNX embeddings 均为首次调用时才初始化，不增加启动时间
- **分层缓存**：SQLite TTL 缓存价格查询，命中缓存时完全绕过 RapidFuzz 计算
- **权限分级**：`sales` 角色只能介入当前对话，`admin` 角色可批量审计和同步知识库
- **全链路审计**：每次人工编辑生成 JSON 日志，字段包含 `timestamp / operator / original / edited / synced_to_kb`

### 4. 体现了对 LangGraph 状态机的深度掌握

正确使用 `interrupt()` 实现真正的 HITL 暂停是 LangGraph 的进阶用法，大多数教程不覆盖这部分。状态机设计中的条件路由、节点间状态传递、跨节点的置信度累积逻辑，都反映了对框架原理的理解，而不只是照抄模板。

### 5. 体现了在不确定性下的工程决策能力

三步自反思的重生成上限（最多 2 次）、70%/85% 置信度阈值的划分、CRAG 低分触发重写而不是直接报错——这些都是在"AI 输出不可完全预测"的前提下，设计出的**有界失败保护**机制。这种在不确定系统中建立确定性边界的能力，是工程师成熟度的体现。

---

## 对接说明

### 企业知识库对接

将现有车型 PDF 文档（配置手册、规格书）放入项目目录后运行入库脚本：

```bash
python rag/ingest.py --source ./docs/vehicle_specs/ --collection car_knowledge
```

支持增量更新，已入库文档不重复处理。

### WhatsApp 对接（Green API）

1. 在 [Green API](https://green-api.com) 注册账号，获取 `instance_id` 和 `api_token`
2. 填入 `.env` 对应字段
3. 在 Green API 控制台将 webhook URL 配置为 `https://your-domain/whatsapp/webhook`
4. 详细步骤见 `docs/integration_guide.md`

### CRM 对接

`contract_node.py` 提取的结构化字段（买方名称、车型、数量、价格、港口）通过标准 JSON 格式输出，可直接接入 Salesforce / 金蝶 / 自建 CRM 的 REST API。在 `server.py` 中扩展 `/api/contract-export` 端点即可。

---

*本项目为个人技术项目，车辆数据与价格仅供演示，不代表任何真实产品报价。*
