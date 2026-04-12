# 汽车出口销售 AI Agent — 对接说明文档

> 版本：v1.0 | 更新日期：2026-04-13 | 技术栈：LangGraph · LangChain · ChromaDB · Streamlit · FastAPI

---

## 目录

1. [系统架构图](#1-系统架构图)
2. [快速部署](#2-快速部署)
3. [对接企业知识库](#3-对接企业知识库)
4. [价格数据对接](#4-价格数据对接)
5. [销售系统对接（API）](#5-销售系统对接api)
6. [反思工作流配置](#6-反思工作流配置)
7. [人工介入与权限管理](#7-人工介入与权限管理)
8. [常见问题排查](#8-常见问题排查)

---

## 1. 系统架构图

### 1.1 整体数据流

```
┌─────────────────────────────────────────────────────────────────────┐
│                          输入渠道层                                   │
│                                                                       │
│   WhatsApp (Green API)          Web 前端 (Streamlit)                 │
│        │                                   │                         │
│        └──────────────┬────────────────────┘                         │
└───────────────────────┼─────────────────────────────────────────────┘
                        │ HTTP POST
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FastAPI 服务层 (server.py)                        │
│                                                                       │
│   POST /chat          POST /whatsapp/webhook                         │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   LangGraph 工作流 (agent/graph.py)                   │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                    AgentState (state.py)                      │    │
│  │  messages / intent / price_result / rag_result /             │    │
│  │  contract / reflection_steps / intervention_needed           │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                        │                                              │
│                        ▼                                              │
│              ┌─────────────────┐                                      │
│              │  意图识别节点    │  intent_detector.py                 │
│              │  (IntentNode)   │  → price_query / rag_query /        │
│              └────────┬────────┘    contract / general               │
│                       │                                               │
│          ┌────────────┼─────────────┐                                │
│          ▼            ▼             ▼                                 │
│   ┌────────────┐ ┌─────────┐ ┌──────────────┐                       │
│   │ 价格查询节点│ │RAG节点  │ │  合同生成节点 │                       │
│   │ price_node │ │rag_node │ │contract_node │                       │
│   │            │ │         │ │              │                        │
│   │ RapidFuzz  │ │ChromaDB │ │  GPT模板填充 │                       │
│   │ 模糊匹配   │ │向量检索 │ │              │                        │
│   │ 区间筛选   │ │         │ │              │                        │
│   │ 二次检索   │ │         │ │              │                        │
│   └─────┬──────┘ └────┬────┘ └──────┬───────┘                       │
│         └─────────────┴─────────────┘                                │
│                        │                                              │
│                        ▼                                              │
│           ┌────────────────────────┐                                  │
│           │   三步自反思流水线       │  reflection_pipeline.py         │
│           │                        │                                  │
│           │  Step 1: 事实核查       │  ← 数据准确性验证               │
│           │  Step 2: 合规检查       │  ← 法规 / 敏感词                │
│           │  Step 3: 追加销售       │  ← 关联产品推荐                 │
│           └────────────┬───────────┘                                  │
│                        │                                              │
│                        ▼                                              │
│           ┌────────────────────────┐                                  │
│           │     人工介入判断        │  human_intervention.py          │
│           │                        │                                  │
│           │  信心度 < 70% ?         │                                  │
│           │  或手动触发 ?           │                                  │
│           └────────┬───────────────┘                                  │
│                    │                                                   │
│         ┌──────────┴──────────┐                                       │
│         ▼                     ▼                                        │
│   ┌───────────┐      ┌────────────────┐                               │
│   │  自动回复  │      │  人工审核队列   │                               │
│   │  (输出)   │      │  interrupt()   │                               │
│   └───────────┘      │  + KB同步      │                               │
│                      └────────────────┘                               │
└─────────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        外部资源层                                      │
│                                                                       │
│   OpenAI 兼容 API          ChromaDB           SQLite TTL 缓存         │
│   hk.ticketpro.cc/v1      (向量存储)          price_cache.db          │
│                                                                       │
│   CSV 价格表               Green API           企业 CRM               │
│   data/prices.csv          WhatsApp 代理       (可选集成)              │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 状态流转说明

| 节点 | 输入 State 字段 | 输出 State 字段 |
|------|----------------|----------------|
| IntentDetector | `messages` | `intent` |
| PriceNode | `intent`, `messages` | `price_result` |
| RAGNode | `intent`, `messages` | `rag_result` |
| ContractNode | `intent`, `messages`, `price_result` | `contract` |
| ReflectionPipeline | `price_result`, `rag_result`, `contract` | `reflection_steps` |
| HumanIntervention | `reflection_steps`, `confidence_score` | `final_response`, `intervention_needed` |

---

## 2. 快速部署

### 2.1 环境要求

| 项目 | 最低要求 |
|------|---------|
| Python | 3.10+ |
| 内存 | 4 GB RAM（ChromaDB ONNX 嵌入模型）|
| 磁盘 | 2 GB（含嵌入模型缓存）|
| 网络 | 能访问 `https://hk.ticketpro.cc/v1`（LLM 代理）|
| 操作系统 | Linux / macOS / Windows（WSL2 推荐）|

### 2.2 克隆与安装

```bash
# 1. 克隆项目
git clone https://github.com/your-org/car-export-agent.git
cd car-export-agent

# 2. 创建虚拟环境（推荐）
python3.10 -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# 3. 安装依赖
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.3 .env 配置说明

在项目根目录创建 `.env` 文件（可复制 `.env.example`）：

```dotenv
# ── LLM 配置 ──────────────────────────────────────────────────────────
# OpenAI 兼容 API 密钥（代理地址）
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_BASE_URL=https://hk.ticketpro.cc/v1
# 使用的模型名称（与代理服务商确认可用模型）
OPENAI_MODEL=gpt-4o

# ── WhatsApp (Green API) ───────────────────────────────────────────────
# 在 greenapi.com 注册后获取
GREEN_API_INSTANCE_ID=1234567890
GREEN_API_ACCESS_TOKEN=your_green_api_token_here
# 本地服务对外地址（Green API 回调用）
WEBHOOK_BASE_URL=https://your-domain.com

# ── ChromaDB 向量存储 ──────────────────────────────────────────────────
CHROMA_PERSIST_DIR=./data/chroma_db
CHROMA_COLLECTION_NAME=car_knowledge_base

# ── 价格缓存 ──────────────────────────────────────────────────────────
PRICE_CACHE_DB=./data/price_cache.db
PRICE_CSV_PATH=./data/prices.csv
# 缓存过期时间（秒），默认 3600（1小时）
PRICE_CACHE_TTL=3600

# ── 人工介入 ──────────────────────────────────────────────────────────
# 信心度阈值（低于此值触发人工介入）
CONFIDENCE_THRESHOLD=0.70
# 审计日志路径
INTERVENTION_LOG_PATH=./data/intervention_log.jsonl

# ── 反思工作流 ─────────────────────────────────────────────────────────
# lenient / normal / strict
REFLECTION_STRICTNESS=normal

# ── FastAPI ───────────────────────────────────────────────────────────
API_HOST=0.0.0.0
API_PORT=8000
# 内部通信密钥（前后端鉴权用）
INTERNAL_API_KEY=change_me_in_production
```

> **安全提示**：`.env` 已加入 `.gitignore`，请勿提交到版本控制系统。

### 2.4 初始化向量库

首次部署需将产品手册 PDF 导入 ChromaDB：

```bash
# 将 PDF 文件放入 data/docs/ 目录，然后执行：
python -m rag.ingest --docs-dir ./data/docs --verbose
# 输出示例：
# [INFO] Loading: Toyota_Hilux_Manual_2024.pdf  (42 pages)
# [INFO] Loading: BYD_Atto3_Export_Spec.pdf     (28 pages)
# [INFO] Chunks created: 847
# [INFO] Embeddings written to ChromaDB: ./data/chroma_db
# [SUCCESS] Ingestion complete.
```

### 2.5 启动服务

**方式一：分别启动（开发环境）**

```bash
# 终端 1 — FastAPI 后端
uvicorn server:app --host 0.0.0.0 --port 8000 --reload

# 终端 2 — Streamlit 前端
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

**方式二：Docker Compose（生产推荐）**

```yaml
# docker-compose.yml
version: "3.9"
services:
  api:
    build: .
    command: uvicorn server:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    env_file: .env
    volumes:
      - ./data:/app/data

  frontend:
    build: .
    command: streamlit run app.py --server.port 8501 --server.address 0.0.0.0
    ports:
      - "8501:8501"
    env_file: .env
    depends_on:
      - api
```

```bash
docker compose up -d
```

**访问地址**

| 服务 | 地址 |
|------|------|
| Streamlit 前端 | http://localhost:8501 |
| FastAPI 文档 | http://localhost:8000/docs |
| FastAPI ReDoc | http://localhost:8000/redoc |

---

## 3. 对接企业知识库

### 3.1 上传 PDF 产品手册到 RAG

#### 文件准备

将以下类型文件放入 `data/docs/` 目录：

```
data/docs/
├── product_manuals/        # 产品手册
│   ├── Toyota_Hilux_2024.pdf
│   └── BYD_Seal_Export.pdf
├── compliance/             # 合规文件
│   ├── Export_Regulations_Africa.pdf
│   └── CE_Certificate_Template.pdf
└── price_lists/            # 报价单（可选，也可用 CSV 方式）
    └── Q2_2026_Pricelist.pdf
```

#### 调用 rag/ingest.py 步骤

```python
# 方式一：命令行调用
python -m rag.ingest \
    --docs-dir ./data/docs \
    --collection-name car_knowledge_base \
    --chunk-size 512 \
    --chunk-overlap 64 \
    --verbose

# 方式二：Python API 调用（集成到上传流程中）
from rag.ingest import ingest_documents

result = ingest_documents(
    docs_dir="./data/docs",
    collection_name="car_knowledge_base",
    chunk_size=512,
    chunk_overlap=64,
)
print(f"已导入 {result['chunks_added']} 个文本块")
```

#### ingest.py 内部流程

```
PDF 文件
    │
    ▼ langchain_community.document_loaders.PyPDFLoader
文档对象列表
    │
    ▼ RecursiveCharacterTextSplitter(chunk_size=512, overlap=64)
文本块列表
    │
    ▼ ChromaDB ONNX 内置嵌入（all-MiniLM-L6-v2）
向量 + 元数据
    │
    ▼ ChromaDB.add_documents()
持久化到 ./data/chroma_db
```

### 3.2 get_vectorstore() 接口说明

```python
# rag/vectorstore.py

def get_vectorstore(
    collection_name: str = None,      # 集合名称，None 时读取 .env
    persist_dir: str = None,          # 持久化目录，None 时读取 .env
    embedding_fn=None,                # 嵌入函数，None 时使用 ONNX 内置
) -> Chroma:
    """
    获取或初始化 ChromaDB 向量存储实例。
    
    返回值：
        langchain_chroma.Chroma 实例，支持以下操作：
          .similarity_search(query, k=5)          → List[Document]
          .similarity_search_with_score(query, k) → List[Tuple[Document, float]]
          .add_documents(docs)                    → List[str]（文档 ID）
          .delete(ids)                            → None
    
    示例：
        vs = get_vectorstore()
        results = vs.similarity_search("BYD Atto3 的续航里程是多少", k=3)
        for doc in results:
            print(doc.page_content)
            print(doc.metadata)  # {"source": "BYD_Atto3.pdf", "page": 12}
    """
```

### 3.3 替换 ChromaDB 为其他向量数据库

系统通过 `get_vectorstore()` 抽象层隔离向量库实现，替换步骤如下：

#### 替换为 Pinecone

```python
# rag/vectorstore.py — 修改 get_vectorstore() 函数

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

def get_vectorstore(collection_name: str = None, **kwargs):
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_BASE_URL"),
    )
    return PineconeVectorStore(
        index_name=collection_name or os.getenv("PINECONE_INDEX"),
        embedding=embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    )
```

#### 替换为 Weaviate

```python
from langchain_weaviate import WeaviateVectorStore
import weaviate

def get_vectorstore(collection_name: str = None, **kwargs):
    client = weaviate.connect_to_wcs(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )
    return WeaviateVectorStore(
        client=client,
        index_name=collection_name or "CarKnowledge",
        text_key="text",
    )
```

#### 替换为企业内部 Milvus

```python
from langchain_milvus import Milvus

def get_vectorstore(collection_name: str = None, **kwargs):
    return Milvus(
        collection_name=collection_name or "car_kb",
        connection_args={
            "host": os.getenv("MILVUS_HOST", "localhost"),
            "port": os.getenv("MILVUS_PORT", "19530"),
        },
        embedding_function=get_default_embedding(),
    )
```

> **注意**：替换向量库后，需同步更新 `.env` 添加对应服务的连接参数，并重新执行 `rag/ingest.py` 将数据导入新的向量库。

### 3.4 增量更新知识库

```python
# 仅追加新文档（不重建整个向量库）
from rag.ingest import ingest_documents

result = ingest_documents(
    docs_dir="./data/docs/new_arrivals",   # 只放新文件
    collection_name="car_knowledge_base",
    mode="append",                          # append | rebuild
)
print(f"新增 {result['chunks_added']} 个文本块")
```

---

## 4. 价格数据对接

### 4.1 prices.csv 列格式说明

`data/prices.csv` 是价格查询的基础数据源，标准列格式如下：

```csv
product_id,brand,model,variant,year,currency,price_usd,price_cny,market,availability,min_order_qty,update_time
CAR-001,Toyota,Hilux,Double Cab 4x4,2024,USD,32500,235000,Africa;Middle East,in_stock,1,2026-04-01T08:00:00Z
CAR-002,Toyota,Hilux,Single Cab 4x2,2024,USD,24800,179000,Africa,in_stock,5,2026-04-01T08:00:00Z
CAR-003,BYD,Atto3,Standard Range,2024,USD,22000,158000,Global,in_stock,1,2026-03-15T10:30:00Z
CAR-004,BYD,Seal,Long Range AWD,2024,USD,38900,280000,Europe;Australia,pre_order,1,2026-03-20T09:00:00Z
CAR-005,Chery,Tiggo8,Pro Max,2024,USD,17500,126000,Africa;Southeast Asia,in_stock,3,2026-04-05T14:00:00Z
```

**字段说明**

| 字段 | 类型 | 说明 | 是否必填 |
|------|------|------|---------|
| `product_id` | string | 产品唯一标识，格式 `CAR-XXX` | 是 |
| `brand` | string | 品牌名称（用于模糊匹配） | 是 |
| `model` | string | 车型名称 | 是 |
| `variant` | string | 具体配置/版本 | 否 |
| `year` | integer | 年款 | 是 |
| `currency` | string | 标价币种（USD/CNY/EUR） | 是 |
| `price_usd` | float | 美元含税出口价 | 是 |
| `price_cny` | float | 人民币价格（内部参考） | 否 |
| `market` | string | 目标市场，多个用 `;` 分隔 | 否 |
| `availability` | string | `in_stock` / `pre_order` / `discontinued` | 是 |
| `min_order_qty` | integer | 最低起订量 | 否 |
| `update_time` | ISO8601 | 价格最后更新时间 | 是 |

### 4.2 添加新产品

**方式一：直接编辑 CSV**

```bash
# 在 data/prices.csv 末尾追加新行
echo 'CAR-006,SAIC,MG,ZS EV Long Range,2024,USD,19900,143000,Global,in_stock,1,2026-04-13T00:00:00Z' \
    >> data/prices.csv

# 刷新缓存（让修改立即生效）
python -c "from agent.utils.price_cache import invalidate_cache; invalidate_cache()"
```

**方式二：通过管理 API 添加**

```bash
curl -X POST http://localhost:8000/admin/prices \
  -H "Authorization: Bearer $INTERNAL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "product_id": "CAR-006",
    "brand": "SAIC",
    "model": "MG",
    "variant": "ZS EV Long Range",
    "year": 2024,
    "currency": "USD",
    "price_usd": 19900,
    "market": "Global",
    "availability": "in_stock",
    "min_order_qty": 1
  }'
```

**方式三：批量导入（外部系统同步）**

```python
import pandas as pd
from agent.utils.price_cache import invalidate_cache

def sync_prices_from_erp(erp_data: list[dict]):
    """
    从 ERP 系统同步价格数据到 prices.csv。
    
    Args:
        erp_data: ERP 系统导出的产品列表（字典格式）
    """
    df_new = pd.DataFrame(erp_data)
    
    # 确保必填字段存在
    required_cols = ["product_id", "brand", "model", "year", "price_usd", "availability"]
    assert all(col in df_new.columns for col in required_cols), "缺少必填字段"
    
    # 添加 update_time
    df_new["update_time"] = pd.Timestamp.utcnow().isoformat()
    
    # 合并到现有 CSV（以 product_id 去重，保留新版本）
    df_existing = pd.read_csv("data/prices.csv")
    df_merged = pd.concat([df_existing, df_new]).drop_duplicates(
        subset=["product_id"], keep="last"
    )
    df_merged.to_csv("data/prices.csv", index=False)
    
    # 使缓存失效，强制下次查询重新加载
    invalidate_cache()
    print(f"同步完成：新增/更新 {len(df_new)} 条记录")
```

### 4.3 价格缓存机制（SQLite TTL）

```
                  查询请求（品牌/型号/价格区间）
                         │
                         ▼
             ┌───────────────────────┐
             │  price_cache.py       │
             │  检查 SQLite 缓存      │
             └───────────┬───────────┘
                         │
             ┌───────────┴───────────┐
             │                       │
         缓存命中                  缓存未命中
     (TTL 未过期)              (首次查询或已过期)
             │                       │
             ▼                       ▼
        直接返回                读取 prices.csv
        缓存结果               RapidFuzz 模糊匹配
             │                       │
             │                  写入 SQLite 缓存
             │                  (附 expired_at 时间戳)
             │                       │
             └───────────┬───────────┘
                         ▼
                    返回查询结果
```

**缓存配置参数（.env）**

```dotenv
PRICE_CACHE_TTL=3600        # 1小时过期（生产建议 1800）
PRICE_CACHE_DB=./data/price_cache.db
```

**手动操作缓存**

```python
from agent.utils.price_cache import (
    get_cached_price,
    set_price_cache,
    invalidate_cache,
    get_cache_stats,
)

# 查看缓存统计
stats = get_cache_stats()
print(stats)
# {"total_entries": 142, "expired": 3, "hit_rate": "87.3%"}

# 强制清空所有缓存（价格大幅调整时使用）
invalidate_cache()

# 清空特定产品的缓存
invalidate_cache(product_id="CAR-001")
```

### 4.4 价格区间查询示例

Agent 自动处理自然语言价格区间查询：

**用户输入示例**

```
用户：我想找 15000 到 25000 美元之间的 SUV，最好是国产品牌
```

**内部处理（price_node.py）**

```python
# 1. 意图识别提取价格区间
intent_payload = {
    "type": "price_query",
    "price_min": 15000,
    "price_max": 25000,
    "currency": "USD",
    "keywords": ["SUV", "国产"],
}

# 2. PriceNode 执行区间筛选 + 模糊匹配
def query_by_range(
    price_min: float,
    price_max: float,
    keywords: list[str],
    df: pd.DataFrame,
) -> pd.DataFrame:
    # 第一步：价格区间硬过滤
    filtered = df[
        (df["price_usd"] >= price_min) & 
        (df["price_usd"] <= price_max) &
        (df["availability"] != "discontinued")
    ].copy()
    
    # 第二步：RapidFuzz 关键词模糊匹配（品牌/型号/variant）
    from rapidfuzz import process, fuzz
    
    def score_row(row):
        search_text = f"{row['brand']} {row['model']} {row.get('variant', '')}"
        scores = [
            fuzz.partial_ratio(kw, search_text) 
            for kw in keywords
        ]
        return max(scores) if scores else 0
    
    filtered["match_score"] = filtered.apply(score_row, axis=1)
    
    # 第三步：如果模糊匹配无结果，降低阈值做二次检索
    threshold = 60
    result = filtered[filtered["match_score"] >= threshold]
    if result.empty:
        threshold = 30
        result = filtered[filtered["match_score"] >= threshold]
    
    return result.sort_values("match_score", ascending=False)

# 3. 返回格式化结果
# [
#   {"product_id": "CAR-005", "brand": "Chery", "model": "Tiggo8",
#    "price_usd": 17500, "availability": "in_stock", "match_score": 75},
#   {"product_id": "CAR-003", "brand": "BYD", "model": "Atto3",
#    "price_usd": 22000, "availability": "in_stock", "match_score": 68},
# ]
```

**API 调用价格区间查询**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "有没有 15000 到 25000 美元的国产 SUV？",
    "session_id": "demo-001",
    "user_role": "sales"
  }'
```

---

## 5. 销售系统对接（API）

### 5.1 FastAPI 端点说明

#### POST /chat — 主对话接口

```
POST http://localhost:8000/chat
Content-Type: application/json
Authorization: Bearer {INTERNAL_API_KEY}
```

**请求体**

```json
{
  "message": "我想了解丰田 Hilux 的出口价格",
  "session_id": "user-uuid-or-whatsapp-number",
  "user_role": "sales",
  "metadata": {
    "channel": "web",
    "language": "zh-CN",
    "customer_id": "CRM-12345"
  }
}
```

**响应体**

```json
{
  "session_id": "user-uuid-or-whatsapp-number",
  "response": "Toyota Hilux Double Cab 4x4（2024款）出口含税价为 **$32,500 USD**，最低起订量 1 辆。Double Cab 4x2 版本售价 $28,000，适合预算有限的市场。请问您主要面向哪个目标市场？",
  "intent": "price_query",
  "confidence": 0.92,
  "reflection_passed": true,
  "intervention_triggered": false,
  "products_mentioned": ["CAR-001", "CAR-002"],
  "processing_time_ms": 1243,
  "timestamp": "2026-04-13T10:30:00Z"
}
```

**需要人工介入时的响应**

```json
{
  "session_id": "user-uuid-or-whatsapp-number",
  "response": null,
  "intent": "price_query",
  "confidence": 0.58,
  "reflection_passed": false,
  "intervention_triggered": true,
  "intervention_id": "INT-20260413-0042",
  "pending_message": "您的问题已转交给我们的专业销售顾问，将在 15 分钟内为您回复。",
  "timestamp": "2026-04-13T10:31:00Z"
}
```

#### POST /whatsapp/webhook — WhatsApp 回调接口

```
POST http://localhost:8000/whatsapp/webhook
Content-Type: application/json
```

**Green API 推送格式（系统自动处理）**

```json
{
  "typeWebhook": "incomingMessageReceived",
  "instanceData": {
    "idInstance": 1234567890,
    "wid": "1234567890@c.us"
  },
  "timestamp": 1712999999,
  "idMessage": "BAE5F4886F532345",
  "senderData": {
    "chatId": "79001234567@c.us",
    "senderName": "Ahmed Al-Rashid"
  },
  "messageData": {
    "typeMessage": "textMessage",
    "textMessageData": {
      "textMessage": "I need price for BYD Atto3, minimum 10 units"
    }
  }
}
```

#### GET /health — 健康检查

```bash
curl http://localhost:8000/health
# {"status": "ok", "chroma_docs": 847, "price_records": 48, "version": "1.0.0"}
```

#### GET /admin/interventions — 获取待审核列表（需管理员权限）

```bash
curl http://localhost:8000/admin/interventions?status=pending \
  -H "Authorization: Bearer $INTERNAL_API_KEY"
```

### 5.2 对接 CRM 系统

以下示例展示如何将 Agent 响应自动同步到 CRM（以 Salesforce / 国内通用 CRM 为例）：

```python
# integrations/crm_sync.py

import httpx
import json
from datetime import datetime


class CRMSyncClient:
    """
    CRM 同步客户端。
    在 Agent 生成响应后，将对话记录、意图、产品兴趣同步到 CRM。
    """
    
    def __init__(self, crm_base_url: str, crm_api_key: str):
        self.base_url = crm_base_url
        self.headers = {
            "Authorization": f"Bearer {crm_api_key}",
            "Content-Type": "application/json",
        }
    
    async def log_interaction(
        self,
        customer_id: str,
        session_id: str,
        message: str,
        response: str,
        intent: str,
        products_mentioned: list[str],
        confidence: float,
    ) -> dict:
        """记录客户交互到 CRM 活动日志"""
        payload = {
            "customerId": customer_id,
            "channel": "ai_agent",
            "sessionId": session_id,
            "direction": "inbound",
            "content": message,
            "agentReply": response,
            "tags": [intent] + products_mentioned,
            "aiConfidence": confidence,
            "timestamp": datetime.utcnow().isoformat(),
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/api/v1/activities",
                headers=self.headers,
                json=payload,
                timeout=5.0,
            )
            return resp.json()
    
    async def update_lead_interest(
        self, customer_id: str, product_ids: list[str]
    ) -> None:
        """更新客户感兴趣的产品（用于销售跟进）"""
        payload = {
            "customerId": customer_id,
            "interestedProducts": product_ids,
            "lastContactedAt": datetime.utcnow().isoformat(),
        }
        async with httpx.AsyncClient() as client:
            await client.patch(
                f"{self.base_url}/api/v1/leads/{customer_id}",
                headers=self.headers,
                json=payload,
                timeout=5.0,
            )


# 在 server.py 中集成
from fastapi import FastAPI
from integrations.crm_sync import CRMSyncClient

app = FastAPI()
crm = CRMSyncClient(
    crm_base_url=os.getenv("CRM_BASE_URL"),
    crm_api_key=os.getenv("CRM_API_KEY"),
)

@app.post("/chat")
async def chat(request: ChatRequest):
    result = await run_agent(request.message, request.session_id)
    
    # 异步同步到 CRM（不阻塞响应）
    if request.metadata.get("customer_id"):
        asyncio.create_task(
            crm.log_interaction(
                customer_id=request.metadata["customer_id"],
                session_id=request.session_id,
                message=request.message,
                response=result["response"],
                intent=result["intent"],
                products_mentioned=result.get("products_mentioned", []),
                confidence=result["confidence"],
            )
        )
    
    return result
```

### 5.3 对接 WhatsApp（Green API）完整流程

#### 步骤一：注册 Green API 并配置 Webhook

1. 访问 [https://greenapi.com](https://greenapi.com) 注册账号
2. 创建实例，扫描 WhatsApp 二维码绑定手机
3. 在实例设置中填写 Webhook URL：
   ```
   https://your-domain.com/whatsapp/webhook
   ```
4. 将 `INSTANCE_ID` 和 `ACCESS_TOKEN` 填入 `.env`

#### 步骤二：配置本地开发环境的 Webhook 穿透

```bash
# 使用 ngrok 暴露本地端口（开发用）
ngrok http 8000
# 将 https://xxxx.ngrok-free.app/whatsapp/webhook 填入 Green API 控制台
```

#### 步骤三：发送消息回复（handler.py 内部实现）

```python
# whatsapp/handler.py

import httpx
import os

GREEN_API_BASE = "https://api.green-api.com"

async def send_whatsapp_message(chat_id: str, message: str) -> bool:
    """
    通过 Green API 发送 WhatsApp 消息。
    
    Args:
        chat_id: WhatsApp 聊天 ID，格式为 "国家码手机号@c.us"
                 例：+86 13800138000 → "8613800138000@c.us"
        message: 要发送的文本内容
    """
    instance_id = os.getenv("GREEN_API_INSTANCE_ID")
    token = os.getenv("GREEN_API_ACCESS_TOKEN")
    
    url = f"{GREEN_API_BASE}/waInstance{instance_id}/sendMessage/{token}"
    payload = {
        "chatId": chat_id,
        "message": message,
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, timeout=10.0)
        return response.status_code == 200
```

---

## 6. 反思工作流配置

### 6.1 三步检查详细说明

`agent/nodes/reflection_pipeline.py` 在正式输出前对 Agent 生成的内容执行三次 LLM 自检：

```
生成的初始回复
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  Step 1: 事实核查 (Fact Check)                       │
│                                                      │
│  检查项目：                                           │
│  ✓ 价格数据是否与 CSV 记录一致                        │
│  ✓ 产品规格是否与 RAG 知识库匹配                      │
│  ✓ 是否存在数字错误（价格单位、数量）                  │
│  ✓ 产品是否仍在销售（availability 检查）              │
│                                                      │
│  error_type: "price_mismatch" | "spec_error" |      │
│             "discontinued_product"                   │
└──────────────────┬──────────────────────────────────┘
                   │ 通过 / 已修正
                   ▼
┌─────────────────────────────────────────────────────┐
│  Step 2: 合规检查 (Compliance Check)                  │
│                                                      │
│  检查项目：                                           │
│  ✓ 是否包含出口管制违禁词                             │
│  ✓ 是否做出不当的价格承诺（"最低价"等绝对表述）        │
│  ✓ 是否涉及未经授权的折扣                             │
│  ✓ 是否符合目标市场的贸易法规                         │
│                                                      │
│  error_type: "compliance_violation" |               │
│             "unauthorized_commitment"                │
└──────────────────┬──────────────────────────────────┘
                   │ 通过 / 已修正
                   ▼
┌─────────────────────────────────────────────────────┐
│  Step 3: 追加销售 (Upsell Suggestion)                 │
│                                                      │
│  检查项目：                                           │
│  ✓ 是否有机会推荐更高价值车型                         │
│  ✓ 是否应推荐配件/延保服务                            │
│  ✓ 是否提及了批量折扣机会                             │
│  ✓ 是否引导下一步动作（询价/签合同）                   │
│                                                      │
│  trigger_condition: "budget_headroom" |             │
│                    "fleet_purchase" | "none"         │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
             最终输出回复
```

### 6.2 反思日志字段说明

每次反思流水线执行后，状态中会包含 `reflection_steps` 字段：

```json
{
  "reflection_steps": [
    {
      "step": 1,
      "step_name": "fact_check",
      "passed": false,
      "error_type": "price_mismatch",
      "original_claim": "Toyota Hilux 售价 $28,500",
      "correction_plan": "将价格修正为 CSV 记录的 $32,500（CAR-001）",
      "corrected": true,
      "latency_ms": 421
    },
    {
      "step": 2,
      "step_name": "compliance_check",
      "passed": true,
      "error_type": null,
      "correction_plan": null,
      "corrected": false,
      "latency_ms": 318
    },
    {
      "step": 3,
      "step_name": "upsell_suggestion",
      "passed": true,
      "trigger_condition": "fleet_purchase",
      "suggestion_added": "已在回复末尾追加：10辆以上可享受2%批量折扣，欢迎联系销售顾问。",
      "latency_ms": 389
    }
  ],
  "total_reflection_latency_ms": 1128,
  "final_confidence_adjustment": +0.08
}
```

**字段说明**

| 字段 | 说明 |
|------|------|
| `step` | 步骤编号（1/2/3） |
| `step_name` | 步骤名称 |
| `passed` | 是否通过检查 |
| `error_type` | 发现的错误类型（通过时为 null） |
| `original_claim` | Agent 原始输出中有问题的内容 |
| `correction_plan` | 修正方案描述 |
| `corrected` | 是否已自动修正 |
| `trigger_condition` | 追加销售的触发条件 |
| `latency_ms` | 该步骤耗时（毫秒） |

### 6.3 严格度配置

在 `.env` 中设置 `REFLECTION_STRICTNESS`：

| 模式 | 适用场景 | 行为说明 |
|------|---------|---------|
| `lenient` | 内部测试 / Demo 演示 | 仅执行 Step 1，跳过合规和追加销售 |
| `normal` | 生产环境（推荐） | 执行全部三步，有容错空间 |
| `strict` | 高合规要求市场（欧洲/北美）| 全部三步执行，任何 warning 都触发人工介入 |

```python
# config/settings.py 中的严格度映射
STRICTNESS_CONFIG = {
    "lenient": {
        "enabled_steps": [1],
        "auto_correct": True,
        "intervention_on_warning": False,
    },
    "normal": {
        "enabled_steps": [1, 2, 3],
        "auto_correct": True,
        "intervention_on_warning": False,
    },
    "strict": {
        "enabled_steps": [1, 2, 3],
        "auto_correct": False,       # 不自动修正，转人工
        "intervention_on_warning": True,
    },
}
```

---

## 7. 人工介入与权限管理

### 7.1 触发条件

人工介入通过 LangGraph 的 `interrupt()` 机制实现，以下任一条件满足时触发：

| 触发类型 | 条件 | 说明 |
|---------|------|------|
| 低信心度 | `confidence_score < 0.70` | 意图识别或 RAG 检索置信度不足 |
| 反思失败 | `reflection_step.corrected == False` | Step 1/2 发现错误但无法自动修正 |
| 严格模式 | `REFLECTION_STRICTNESS=strict` + 任何警告 | 见 6.3 节 |
| 手动触发 | 销售人员在 UI 点击"转人工"按钮 | 前端直接调用 `/chat/escalate` 接口 |
| 合同请求 | `intent == "contract"` + 金额 > 阈值 | 大额合同强制人工审核 |

### 7.2 销售角色 vs 管理员角色

| 权限 | 销售 (sales) | 管理员 (admin) |
|------|:----------:|:------------:|
| 查看对话历史 | 自己的 | 所有 |
| 手动触发人工介入 | ✓ | ✓ |
| 审核待介入队列 | ✓（自己负责的）| ✓（全部）|
| 编辑回复内容 | ✓ | ✓ |
| 批准/拒绝内容同步到知识库 | ✗ | ✓ |
| 修改反思严格度 | ✗ | ✓ |
| 查看审计日志 | ✗ | ✓ |
| 管理 API 密钥 | ✗ | ✓ |

在请求中通过 `user_role` 字段传递角色：

```json
{"message": "...", "session_id": "...", "user_role": "admin"}
```

### 7.3 Admin 批量审核操作指南

#### 查看待审核列表

```bash
# 获取所有待审核的介入请求
curl http://localhost:8000/admin/interventions?status=pending&limit=20 \
  -H "Authorization: Bearer $INTERNAL_API_KEY"

# 响应示例
{
  "total": 5,
  "items": [
    {
      "intervention_id": "INT-20260413-0042",
      "session_id": "8613800138000@c.us",
      "customer_name": "Ahmed Al-Rashid",
      "original_message": "What is the best price for 50 units Hilux?",
      "agent_draft": "Toyota Hilux 50辆批量价约 $28.5/辆...",
      "trigger_reason": "low_confidence",
      "confidence_score": 0.58,
      "created_at": "2026-04-13T10:31:00Z",
      "assigned_to": null
    }
  ]
}
```

#### 批量处理审核

```bash
# 审核并批准（发送修改后的内容）
curl -X POST http://localhost:8000/admin/interventions/INT-20260413-0042/approve \
  -H "Authorization: Bearer $INTERNAL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "edited_response": "Toyota Hilux Double Cab 4x4 批量50辆报价：单价 $31,200（较单辆价优惠4%），总价 $1,560,000 CIF。请问交货港口偏好哪里？",
    "sync_to_kb": true,
    "kb_note": "50辆以上批量折扣参考价"
  }'

# 拒绝并退回（重新走 Agent 流程）
curl -X POST http://localhost:8000/admin/interventions/INT-20260413-0042/reject \
  -H "Authorization: Bearer $INTERNAL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"reason": "需要重新核算批量运费"}'
```

#### Streamlit 管理界面（内置）

访问 `http://localhost:8501`，切换到「管理员」角色后可看到：

- **介入审核面板**：实时显示待处理队列，支持批量操作
- **知识库管理**：查看/删除已同步的内容片段
- **审计日志**：按日期/操作人筛选操作记录

### 7.4 编辑内容同步到知识库的流程

当管理员在审核时勾选 `sync_to_kb: true`，执行以下流程：

```
Admin 批准 + 勾选"同步到知识库"
             │
             ▼
   human_intervention.py
   extract_knowledge_chunk()
             │
   ┌─────────▼──────────┐
   │  构造 Document 对象  │
   │  {                  │
   │    page_content:    │
   │      edited_response│
   │    metadata: {      │
   │      source: "human"│
   │      approved_by:   │
   │        "admin_id"   │
   │      session_id:    │
   │      timestamp:     │
   │      kb_note:       │
   │    }                │
   │  }                  │
   └─────────┬──────────┘
             │
             ▼
   rag/vectorstore.py
   get_vectorstore().add_documents([doc])
             │
             ▼
   ChromaDB 持久化
   (下次查询即可检索到)
             │
             ▼
   intervention_log.py
   记录审计日志（JSONL）
```

**审计日志格式（data/intervention_log.jsonl）**

```jsonl
{"event":"approved","intervention_id":"INT-20260413-0042","admin_id":"admin_001","session_id":"8613800138000@c.us","confidence_before":0.58,"sync_to_kb":true,"timestamp":"2026-04-13T10:45:00Z"}
{"event":"kb_synced","intervention_id":"INT-20260413-0042","doc_id":"chroma-uuid-xxxx","chunk_preview":"Toyota Hilux Double Cab 4x4 批量50辆报价...","timestamp":"2026-04-13T10:45:01Z"}
```

---

## 8. 常见问题排查

### 8.1 向量库为空时的处理

**症状**：RAG 节点返回空结果，Agent 仅依靠 LLM 参数知识回复，知识准确性下降。

**排查步骤**

```bash
# 步骤 1：检查 ChromaDB 是否有数据
python -c "
from rag.vectorstore import get_vectorstore
vs = get_vectorstore()
collection = vs._collection
print(f'文档数量: {collection.count()}')
"

# 步骤 2：若为 0，重新导入文档
python -m rag.ingest --docs-dir ./data/docs --verbose

# 步骤 3：验证导入是否成功
python -c "
from rag.vectorstore import get_vectorstore
vs = get_vectorstore()
results = vs.similarity_search('Toyota Hilux', k=2)
print(f'检索到 {len(results)} 条结果')
for r in results:
    print(' -', r.page_content[:80])
"
```

**自动回退机制**

RAG 节点在向量库为空或检索置信度 < 0.5 时，会自动降级为 LLM 直接回答，并在响应中添加免责说明：

```python
# agent/nodes/rag_node.py
if not results or max(score for _, score in results_with_score) < 0.5:
    state["rag_result"] = {
        "source": "llm_fallback",
        "disclaimer": "以下信息基于通用知识，请以最新产品手册为准。",
        "content": llm_direct_answer,
    }
```

### 8.2 价格匹配失败的处理

**症状**：用户询问某车型价格，Agent 回复"未找到相关价格信息"。

**排查步骤**

```bash
# 步骤 1：确认 CSV 文件格式
python -c "
import pandas as pd
df = pd.read_csv('data/prices.csv')
print(df.dtypes)
print(df.head(3))
"

# 步骤 2：手动测试模糊匹配
python -c "
from rapidfuzz import process, fuzz
import pandas as pd

df = pd.read_csv('data/prices.csv')
query = 'hilux'  # 用户输入的关键词

candidates = (df['brand'] + ' ' + df['model']).tolist()
matches = process.extract(query, candidates, scorer=fuzz.partial_ratio, limit=5)
for match, score, idx in matches:
    print(f'匹配: {match}, 分数: {score}')
"

# 步骤 3：如果分数 < 60，检查是否有拼写问题或添加别名映射
```

**添加产品别名**（解决常见拼写变体）

```python
# config/settings.py
PRODUCT_ALIASES = {
    "hilux": ["hilux", "hi-lux", "hi lux"],
    "atto3": ["atto3", "atto 3", "atto-3", "海豚"],
    "tiggo": ["tiggo", "tiigo", "奇瑞"],
}
```

**价格缓存问题**

```bash
# 如果价格已更新但 Agent 仍返回旧价格，清空缓存
python -c "from agent.utils.price_cache import invalidate_cache; invalidate_cache(); print('缓存已清空')"
```

### 8.3 LLM API 调用失败的回退机制

**常见错误类型及处理**

| 错误 | HTTP 状态码 | 系统行为 |
|------|-----------|---------|
| API 密钥无效 | 401 | 返回错误提示，触发人工介入 |
| 速率限制 | 429 | 指数退避重试（最多3次），超时后转人工 |
| 代理服务不可用 | 502/503 | 触发人工介入，发送等待通知 |
| 请求超时 | timeout | 重试1次，失败后转人工 |

**重试逻辑（内置于 config/settings.py）**

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=False,
)
async def call_llm_with_retry(prompt: str) -> str:
    """带指数退避的 LLM 调用"""
    try:
        response = await llm.ainvoke(prompt)
        return response.content
    except Exception as e:
        logger.warning(f"LLM 调用失败，准备重试: {e}")
        raise
```

**全量降级（API 完全不可用时）**

当 LLM API 连续失败超过阈值，系统自动切换到规则引擎模式：

```python
# agent/graph.py 中的降级判断
if state.get("llm_failure_count", 0) >= 3:
    # 降级：仅使用价格 CSV + 预定义模板回复
    state["degraded_mode"] = True
    state["response"] = generate_template_response(state["intent"], state["price_result"])
    state["intervention_needed"] = True  # 同时通知管理员
```

**诊断命令**

```bash
# 测试 LLM API 连通性
python -c "
import os, httpx
resp = httpx.post(
    os.getenv('OPENAI_BASE_URL') + '/chat/completions',
    headers={'Authorization': 'Bearer ' + os.getenv('OPENAI_API_KEY')},
    json={'model': os.getenv('OPENAI_MODEL','gpt-4o'), 'messages': [{'role':'user','content':'ping'}], 'max_tokens': 5},
    timeout=10,
)
print(resp.status_code, resp.json())
"

# 检查环境变量加载是否正常
python -c "
from config.settings import settings
print('API URL:', settings.OPENAI_BASE_URL)
print('Model:', settings.OPENAI_MODEL)
print('API Key 前8位:', settings.OPENAI_API_KEY[:8] + '...')
"
```

### 8.4 其他常见问题速查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| ChromaDB 初始化报错 | ONNX 模型下载失败（网络问题） | 手动下载模型或配置代理 |
| Green API webhook 不触发 | 本地服务未对外暴露 | 使用 ngrok 或部署到公网 |
| 合同生成乱码 | 模板编码问题 | 确保 `contract_node.py` 使用 UTF-8 读取模板 |
| Streamlit 刷新后丢失会话 | session_state 未持久化 | 使用 `st.session_state` 显式保存 |
| 反思耗时过长（>5s） | 每步独立调用 LLM 3次 | 切换到 `lenient` 模式或升级到支持并发的部署方案 |

---

## 附录

### A. 完整 .env 模板

```dotenv
# ── LLM 配置
OPENAI_API_KEY=
OPENAI_BASE_URL=https://hk.ticketpro.cc/v1
OPENAI_MODEL=gpt-4o

# ── WhatsApp
GREEN_API_INSTANCE_ID=
GREEN_API_ACCESS_TOKEN=
WEBHOOK_BASE_URL=

# ── 向量库
CHROMA_PERSIST_DIR=./data/chroma_db
CHROMA_COLLECTION_NAME=car_knowledge_base

# ── 价格
PRICE_CACHE_DB=./data/price_cache.db
PRICE_CSV_PATH=./data/prices.csv
PRICE_CACHE_TTL=3600

# ── 介入
CONFIDENCE_THRESHOLD=0.70
INTERVENTION_LOG_PATH=./data/intervention_log.jsonl

# ── 反思
REFLECTION_STRICTNESS=normal

# ── API 服务
API_HOST=0.0.0.0
API_PORT=8000
INTERNAL_API_KEY=

# ── CRM（可选）
CRM_BASE_URL=
CRM_API_KEY=
```

### B. requirements.txt 关键依赖版本参考

```
langchain>=0.2.0
langgraph>=0.1.0
langchain-community>=0.2.0
langchain-chroma>=0.1.0
chromadb>=0.5.0
streamlit>=1.35.0
fastapi>=0.111.0
uvicorn>=0.30.0
rapidfuzz>=3.9.0
python-dotenv>=1.0.0
pandas>=2.2.0
httpx>=0.27.0
tenacity>=8.3.0
```

### C. 快速集成检查清单

部署前请逐项确认：

- [ ] `.env` 已配置所有必填字段
- [ ] `data/docs/` 目录有 PDF 文档并已执行 `rag.ingest`
- [ ] `data/prices.csv` 有有效产品数据
- [ ] LLM API 连通性测试通过（见 8.3 节诊断命令）
- [ ] FastAPI 健康检查返回 `{"status": "ok"}`
- [ ] Green API Webhook URL 已配置并测试
- [ ] 管理员账号已创建并测试人工介入流程
- [ ] 反思严格度已根据业务需求设置

---

*本文档由汽车出口销售 AI Agent 项目团队维护。如有问题请提交 Issue 或联系项目负责人。*
