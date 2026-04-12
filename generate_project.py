"""
car-export-agent 项目代码生成器
使用 GPT-5.4 生成所有项目文件

运行方式: python generate_project.py
"""

import os
import time
from pathlib import Path
from openai import OpenAI

# ============ 配置 ============
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-b9ede3798a406b316e96984687f3040d2ac80a723b3cd3681a4d9d776c283336")
BASE_URL = "https://hk.ticketpro.cc/v1"  # 代理服务 base URL
MODEL = "gpt-5.4"
MODEL_FALLBACK = "gpt-4.1"
PROJECT_ROOT = Path(__file__).parent

client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)

SYSTEM_PROMPT = """You are an expert Python developer specializing in LangChain, LangGraph, and Streamlit.
Generate complete, production-ready Python code with NO placeholders, NO TODOs, and NO incomplete sections.
Every function must be fully implemented. Include all necessary imports.
Use type hints. Keep code clean and readable.
For Chinese car export domain: prices in USD, FOB/CIF Incoterms, Chinese car brands (BYD, Chery, MG, Geely, SAIC).
"""

# ============ 文件生成任务列表 ============
FILES_TO_GENERATE = [
    {
        "path": "requirements.txt",
        "prompt": """Generate a requirements.txt for a LangGraph + LangChain + Streamlit project.
Include: langgraph>=0.2, langchain>=0.3, langchain-openai, langchain-anthropic, langchain-community,
streamlit>=1.40, chromadb>=0.5, sentence-transformers>=3.0, pandas, pypdf>=4.0,
python-dotenv, pydantic>=2.0.
Pin to stable versions. One package per line."""
    },
    {
        "path": "config/settings.py",
        "prompt": """Generate config/settings.py for a car export AI agent.
Requirements:
- Load API keys from .env (ANTHROPIC_API_KEY, OPENAI_API_KEY)
- Create LLM instance using langchain_anthropic ChatAnthropic (claude-sonnet-4-6 model)
- Create embeddings using HuggingFaceEmbeddings (model: all-MiniLM-L6-v2, device: cpu)
- Export: llm (ChatAnthropic), embeddings (HuggingFaceEmbeddings), PROJECT_ROOT (Path)
- CHROMA_DB_PATH = PROJECT_ROOT / "rag" / "chroma_db"
- PRICES_CSV_PATH = PROJECT_ROOT / "data" / "prices.csv"
- CONTRACTS_OUTPUT_DIR = PROJECT_ROOT / "contracts" / "output"
- CONTRACTS_TEMPLATE_PATH = PROJECT_ROOT / "contracts" / "templates" / "quote_template.md"
- All paths created with mkdir(parents=True, exist_ok=True)"""
    },
    {
        "path": "config/prompts.py",
        "prompt": """Generate config/prompts.py containing all system prompts as Python string constants.
Include:
1. INTENT_DETECTION_PROMPT: few-shot prompt to classify user message into exactly one of:
   "price_query", "product_info", "contract_request", "general_chat".
   Include 3 examples for each intent. Return ONLY the intent label, nothing else.
2. PRICE_ANSWER_PROMPT: format price query results into a professional sales response.
   Include FOB/CIF prices, min order qty, payment terms suggestion.
3. RAG_ANSWER_PROMPT: answer product questions using retrieved context only.
   Cite the source document. If context doesn't contain answer, say so honestly.
4. CONTRACT_EXTRACT_PROMPT: extract from conversation: buyer_company, buyer_country,
   car_model, car_brand, quantity, destination_port. Return as JSON.
5. REFLECTION_PROMPT: evaluate answer quality (0-10 score) based on:
   - factual accuracy (is it supported by retrieved context?)
   - completeness (did it answer the question?)
   - hallucination risk (did it make up specific numbers?)
   Return JSON: {"score": int, "reason": str, "needs_retry": bool}
6. GENERAL_CHAT_PROMPT: friendly car export sales assistant persona."""
    },
    {
        "path": "agent/state.py",
        "prompt": """Generate agent/state.py with AgentState TypedDict for LangGraph.
Fields:
- messages: Annotated[list, add_messages]  # conversation history
- intent: str  # price_query/product_info/contract_request/general_chat/unclear
- retrieved_context: str  # RAG retrieved text
- price_result: dict  # price lookup result from CSV
- draft_answer: str  # LLM draft response
- reflection_score: int  # 0-10 quality score
- reflection_count: int  # retry counter (max 2)
- needs_retry: bool  # set by reflector node
- contract_data: dict  # extracted contract fields
- contract_path: str  # path to generated contract file
- agent_steps: list  # list of step descriptions for UI display

Import from: langgraph.graph.message import add_messages, typing_extensions.TypedDict
Include a function get_default_state() that returns dict with all fields initialized to empty/default values."""
    },
    {
        "path": "agent/__init__.py",
        "prompt": "Generate empty __init__.py for agent package."
    },
    {
        "path": "agent/nodes/__init__.py",
        "prompt": "Generate empty __init__.py for agent/nodes package."
    },
    {
        "path": "agent/nodes/intent_detector.py",
        "prompt": """Generate agent/nodes/intent_detector.py for LangGraph.
Function: detect_intent(state: AgentState) -> dict
- Extracts last user message from state["messages"]
- Calls llm (from config.settings) with INTENT_DETECTION_PROMPT
- Returns {"intent": str, "agent_steps": [...existing + "Intent detected: {intent}"]}
- If LLM returns unexpected value, default to "general_chat"
- Import AgentState from agent.state, llm from config.settings, INTENT_DETECTION_PROMPT from config.prompts
- Use HumanMessage and SystemMessage from langchain_core.messages"""
    },
    {
        "path": "agent/nodes/price_node.py",
        "prompt": """Generate agent/nodes/price_node.py for LangGraph.
Function: query_price(state: AgentState) -> dict
Logic:
- Load data/prices.csv using pandas
- Extract car model/brand from last user message using simple keyword matching
  (check if any model_name or brand in CSV appears in the user message, case-insensitive)
- If found: return matching rows as price_result dict
- If not found: return price_result with all cars as a list (show catalog)
- Generate draft_answer using llm + PRICE_ANSWER_PROMPT with the price data
- Update agent_steps with "Price lookup completed"
Import: PRICES_CSV_PATH from config.settings, llm from config.settings"""
    },
    {
        "path": "agent/nodes/rag_node.py",
        "prompt": """Generate agent/nodes/rag_node.py for LangGraph.
Function: retrieve_and_answer(state: AgentState) -> dict
Logic:
- Get last user message
- Use vectorstore from rag.vectorstore to search top 3 relevant chunks
- Combine chunks into retrieved_context string with source metadata
- Call llm with RAG_ANSWER_PROMPT + retrieved_context to generate draft_answer
- Update agent_steps with "RAG retrieved {n} chunks from {sources}"
- If vectorstore not initialized, return draft_answer saying "Product manual not loaded yet"
Import: get_vectorstore from rag.vectorstore, llm from config.settings, RAG_ANSWER_PROMPT from config.prompts"""
    },
    {
        "path": "agent/nodes/contract_node.py",
        "prompt": """Generate agent/nodes/contract_node.py for LangGraph.
Function: generate_contract(state: AgentState) -> dict
Logic:
1. Extract contract fields from conversation using llm + CONTRACT_EXTRACT_PROMPT
   Fields: buyer_company, buyer_country, car_model, car_brand, quantity, destination_port
2. Look up price from prices.csv for the car model
3. Calculate total_amount = unit_price * quantity
4. Generate quote_number as "QT-{YYYYMMDD}-{random 3-digit}"
5. Read quote_template.md from CONTRACTS_TEMPLATE_PATH
6. Fill template with extracted data using str.format() or .replace()
7. Save filled contract to CONTRACTS_OUTPUT_DIR/{quote_number}.md
8. Set contract_path = saved file path
9. Set draft_answer = "Quotation {quote_number} has been generated. Total: USD {total:,.0f}"
10. Update agent_steps
Import all from config.settings, CONTRACTS_TEMPLATE_PATH, PRICES_CSV_PATH"""
    },
    {
        "path": "agent/nodes/reflector.py",
        "prompt": """Generate agent/nodes/reflector.py for LangGraph.
Function: reflect_on_answer(state: AgentState) -> dict
Logic:
1. If reflection_count >= 2: set needs_retry=False, return (max retries reached)
2. Build evaluation prompt combining: original question + draft_answer + retrieved_context
3. Call llm with REFLECTION_PROMPT
4. Parse JSON response: {"score": int, "reason": str, "needs_retry": bool}
5. Update reflection_score with score from response
6. If score < 6: set needs_retry=True, increment reflection_count
7. Else: set needs_retry=False
8. Append to agent_steps: "Reflection score: {score}/10 - {reason}"
9. Use try/except for JSON parse, default to score=7 if parsing fails
Return dict with updated state fields."""
    },
    {
        "path": "agent/nodes/general_chat_node.py",
        "prompt": """Generate agent/nodes/general_chat_node.py for LangGraph.
Function: general_chat(state: AgentState) -> dict
Logic:
- Call llm with GENERAL_CHAT_PROMPT + conversation history
- Set draft_answer = response
- Update agent_steps with "General chat response generated"
Keep response friendly and professional, stay in car export sales persona."""
    },
    {
        "path": "agent/graph.py",
        "prompt": """Generate agent/graph.py - the main LangGraph state machine.

Import all node functions from agent.nodes.*
Import AgentState from agent.state

Build graph:
1. StateGraph(AgentState)
2. Add nodes: "intent_detector", "price_node", "rag_node", "contract_node",
   "general_chat_node", "reflector", "respond"
3. respond node: simple function that takes draft_answer and appends it as AIMessage to messages
4. Add edges:
   - START -> intent_detector
   - intent_detector -> router (conditional edge)
   - router function: returns node name based on state["intent"]
     price_query -> price_node
     product_info -> rag_node
     contract_request -> contract_node
     general_chat -> general_chat_node
     default -> general_chat_node
   - price_node -> reflector
   - rag_node -> reflector
   - contract_node -> reflector
   - general_chat_node -> reflector
   - reflector -> retry_router (conditional edge)
     retry_router: if state["needs_retry"]: return based on original intent
     else: return "respond"
   - respond -> END

5. Compile graph: app = graph.compile()
6. Export: app (compiled graph)

Function: run_agent(user_message: str, chat_history: list) -> tuple[str, list, dict]
- Returns (response_text, updated_agent_steps, contract_info)
- contract_info = {"path": contract_path, "quote_number": ...} if contract was generated"""
    },
    {
        "path": "rag/__init__.py",
        "prompt": "Generate empty __init__.py for rag package."
    },
    {
        "path": "rag/vectorstore.py",
        "prompt": """Generate rag/vectorstore.py for ChromaDB vector store operations.
Functions:
1. get_vectorstore() -> Chroma: loads or creates ChromaDB at CHROMA_DB_PATH
   - Use embeddings from config.settings
   - Collection name: "car_documents"

2. search_documents(query: str, k: int = 3) -> list[dict]:
   - Search vectorstore for top-k similar chunks
   - Return list of: {"content": str, "source": str, "page": int}
   - Handle case where vectorstore is empty (return empty list)

3. is_vectorstore_ready() -> bool:
   - Return True if vectorstore has documents, False otherwise

Import: embeddings, CHROMA_DB_PATH from config.settings
Use: langchain_community.vectorstores.Chroma"""
    },
    {
        "path": "rag/ingest.py",
        "prompt": """Generate rag/ingest.py - document ingestion script.
Functions:
1. load_pdfs(docs_dir: Path) -> list[Document]:
   - Load all .pdf files from docs_dir using PyPDFLoader
   - Print filename as loading
   - Return list of LangChain Documents

2. split_documents(docs: list) -> list[Document]:
   - Use RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
   - Return chunks

3. ingest_documents(docs_dir: Path = None) -> int:
   - If no docs_dir, use PROJECT_ROOT / "data" / "docs"
   - Load and split PDFs
   - If no PDFs found: create a sample text document with car export knowledge
     (basic info about BYD, Chery, MG, Geely brands and their popular models)
   - Create/update ChromaDB vectorstore
   - Return number of chunks ingested

if __name__ == "__main__":
    count = ingest_documents()
    print(f"Ingested {count} document chunks")

Import: embeddings, CHROMA_DB_PATH, PROJECT_ROOT from config.settings"""
    },
    {
        "path": "data/prices.csv",
        "prompt": """Generate a CSV file with car pricing data for Chinese car exports.
Columns: model_name,brand,variant,fob_price_usd,cif_price_usd,currency,year,available,min_order_qty,notes
Include exactly 10 rows with these cars:
1. BYD Atto 3, Standard Range, 18500, 19800
2. BYD Seal, Long Range AWD, 24000, 25500
3. BYD Han EV, 600km range, 28000, 29800
4. BYD Dolphin, Standard, 14000, 15200
5. Chery Tiggo 7 Pro, 2.0T, 14500, 15800
6. Chery Tiggo 8, 7-seat, 16800, 18200
7. MG4 Electric, Standard, 15200, 16400
8. MG ZS EV, Long Range, 17500, 18900
9. Geely Coolray, 1.5T, 13800, 15000
10. SAIC Maxus T90, 4WD Diesel, 22000, 23500
All USD, year 2024, available true. Output ONLY the CSV content, no markdown."""
    },
    {
        "path": "contracts/templates/quote_template.md",
        "prompt": """Generate a professional car export quotation template in Markdown format.
Use these placeholders: {quote_number}, {date}, {valid_until}, {buyer_company}, {buyer_country},
{buyer_contact}, {car_brand}, {car_model}, {car_year}, {car_variant}, {quantity},
{fob_price}, {cif_price}, {destination_port}, {total_amount}, {delivery_time}, {incoterms}

Include sections:
- Header: QUOTATION / 报价单 with quote number and date
- Seller info: Sino Auto Export Co., Ltd., Shanghai
- Buyer info section
- Vehicle Details table
- Payment Terms (30% deposit, 70% before shipment T/T)
- Validity and delivery terms
- Footer noting AI-generated for reference

Make it look professional and suitable for international trade."""
    },
    {
        "path": "app.py",
        "prompt": """Generate app.py - the main Streamlit application.

Layout:
- Wide layout, title "🚗 Car Export AI Agent" subtitle "汽车出口智能销售助理"
- Left sidebar:
  * "Agent Reasoning Steps" section showing st.session_state.agent_steps list
  * Each step as expandable item with icon (🔍 for intent, 💰 for price, 📚 for RAG, 📝 for contract, 🪞 for reflection)
  * If contract was generated: "📥 Download Quotation" button to download the .md file
  * Footer with tech stack info
- Main area: chat interface using st.chat_message

Session state:
- messages: list of {"role": "user"/"assistant", "content": str}
- agent_steps: list of step strings
- last_contract_path: str or None

Chat behavior:
1. Show all messages from session_state.messages
2. User input via st.chat_input("Ask about car prices, specs, or request a quotation...")
3. On submit:
   - Append user message to history
   - Show user message immediately
   - Call run_agent(user_message, messages_history) from agent.graph
   - Show assistant response in chat
   - Update agent_steps in sidebar
   - If contract_path returned, store in last_contract_path

Import: from agent.graph import run_agent
Add a spinner ("Thinking...") while agent is running.

At the bottom of sidebar, add a demo hint:
"💡 Try: 'BYD Atto 3 export price to Malaysia?'"

Include proper error handling - if run_agent fails, show error in chat."""
    },
]


def call_gpt(prompt: str, model: str = MODEL) -> str:
    """Call GPT API to generate code."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=4096,
        )
        return response.choices[0].message.content
    except Exception as e:
        if MODEL_FALLBACK and model != MODEL_FALLBACK:
            print(f"  ⚠️  {model} failed ({e}), trying {MODEL_FALLBACK}...")
            return call_gpt(prompt, MODEL_FALLBACK)
        raise


def generate_file(file_info: dict, project_root: Path) -> bool:
    """Generate a single file using GPT."""
    file_path = project_root / file_info["path"]
    file_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[GEN] Generating: {file_info['path']}")

    try:
        content = call_gpt(file_info["prompt"])

        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first line (```python or ```) and last line (```)
            content = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

        file_path.write_text(content, encoding="utf-8")
        print(f"  [OK] Saved to {file_path}")
        return True

    except Exception as e:
        print(f"  [ERR] Error: {e}")
        return False


def main():
    print("=" * 60)
    print("Car Export Agent - Project Code Generator")
    print(f"Output: {PROJECT_ROOT}")
    print(f"Model: {MODEL}")
    print("=" * 60)

    # Also create a .env file with the API keys
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        env_path.write_text(
            f'ANTHROPIC_API_KEY=your_anthropic_key_here\n'
            f'OPENAI_API_KEY={OPENAI_API_KEY}\n',
            encoding="utf-8"
        )
        print(f"\n📋 Created .env template at {env_path}")

    # Generate all files
    success_count = 0
    fail_count = 0

    for i, file_info in enumerate(FILES_TO_GENERATE, 1):
        print(f"\n[{i}/{len(FILES_TO_GENERATE)}]", end="", flush=True)
        if generate_file(file_info, PROJECT_ROOT):
            success_count += 1
        else:
            fail_count += 1
        # Small delay to avoid rate limiting
        time.sleep(0.5)

    print("\n" + "=" * 60)
    print(f"✅ Generated: {success_count} files")
    if fail_count > 0:
        print(f"❌ Failed: {fail_count} files")
    print("\nNext steps:")
    print("1. Edit .env and add your ANTHROPIC_API_KEY")
    print("2. pip install -r requirements.txt")
    print("3. python rag/ingest.py")
    print("4. streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
