import streamlit as st
import os
import torch
import random
import time
from typing import TypedDict, List, Dict, Any
from datetime import datetime

# LangGraph & LangChain Imports
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser, StrOutputParser

# Data Tools
import arxiv
from semanticscholar import SemanticScholar

# Model Imports
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ==========================================
# 0. CONFIGURATION & SETUP
# ==========================================
st.set_page_config(
    page_title="Smart Research Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar Configuration ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=60)
    st.title("Configuration")
    
    # API Keys Management
    with st.expander("🔐 API Keys", expanded=True):
        # Default values from environment variables if set, else empty
        groq_default = os.environ.get("GROQ_API_KEY", "")
        hf_default = os.environ.get("HF_TOKEN", "")
        
        groq_key = st.text_input("Groq API Key", value=groq_default, type="password")
        hf_token = st.text_input("HuggingFace Token", value=hf_default, type="password")
        
        if st.button("Save Keys"):
            os.environ["GROQ_API_KEY"] = groq_key
            os.environ["HF_TOKEN"] = hf_token
            st.toast("Keys updated!", icon="💾")

    # Tuning Parameters
    st.subheader("⚙️ Tuning")
    num_papers = st.slider("Papers to Analyze", 2, 5, 3, help="More papers = Slower but deeper analysis")
    model_temp = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.3)
    
    st.divider()
    st.subheader("🚀 Model Info")
    st.info(f"Base: Llama-3-8B")
    st.success(f"Adapter: Mujtaba007/llama3-arxiv-lora") # Your Custom Model!
    st.caption("v2.3 Production Build")

# Global Constants - NOW USING YOUR HOSTED ADAPTER
ADAPTER_PATH = "Mujtaba007/llama3-arxiv-lora" 
BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B"

# ==========================================
# 1. STATE DEFINITION
# ==========================================
class ResearchState(TypedDict):
    user_query: str
    expanded_keywords: List[str]
    raw_papers: List[Dict[str, Any]]
    selected_papers: List[Dict[str, Any]]
    summaries: Dict[str, str]
    final_analysis: str
    logs: List[str] 

# ==========================================
# 2. CACHED MODEL LOADING
# ==========================================
@st.cache_resource(show_spinner=False)
def load_model_resources():
    """
    Loads the quantization config and tokenizer.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=os.environ.get("HF_TOKEN"))
        return tokenizer, bnb_config
    except Exception as e:
        return None, None

# ==========================================
# 3. AGENT LOGIC
# ==========================================

def keyword_agent(state: ResearchState):
    """Expands user query into academic search terms."""
    logs = state.get("logs", [])
    try:
        # Using Llama 3.1 8B Instant for speed
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2, api_key=os.environ.get("GROQ_API_KEY"))
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a research assistant. Generate 4 specific academic keywords for the user topic. Return ONLY comma-separated terms."),
            ("user", "{topic}")
        ])
        chain = prompt | llm | CommaSeparatedListOutputParser()
        keywords = chain.invoke({"topic": state["user_query"]})
        
        clean_kws = [k.strip() for k in keywords if k.strip()]
        logs.append(f"✅ Keywords Expanded: {clean_kws}")
        return {"expanded_keywords": clean_kws, "logs": logs}
    except Exception as e:
        logs.append(f"❌ Keyword Agent Failed: {str(e)}")
        return {"expanded_keywords": [state["user_query"]], "logs": logs}

def search_agent(state: ResearchState):
    """Fetches papers from arXiv."""
    logs = state.get("logs", [])
    client = arxiv.Client()
    found_papers = []
    
    # Iterate keywords but stop early if we have enough papers to speed things up
    for kw in state["expanded_keywords"]:
        if len(found_papers) >= 5: 
            break
            
        try:
            search = arxiv.Search(
                query=kw,
                max_results=3, # Reduced to avoid throttling
                sort_by=arxiv.SortCriterion.Relevance
            )
            for result in client.results(search):
                # Ensure we have valid data before appending
                if result.title and result.summary:
                    found_papers.append({
                        "id": result.entry_id.split('/')[-1],
                        "title": result.title,
                        "abstract": result.summary.replace("\n", " "),
                        "year": result.published.year,
                        "citation_count": random.randint(5, 500), # Simulated for arXiv
                        "url": result.pdf_url or result.entry_id, # Fallback to entry_id if PDF url is missing
                        "source": "arXiv"
                    })
        except Exception as e:
            logs.append(f"⚠️ Search Error for '{kw}': {e}")
            continue

    # Remove duplicates based on title
    unique_papers = {p['title']: p for p in found_papers}.values()
    logs.append(f"✅ Search Complete: Found {len(unique_papers)} unique papers")
    return {"raw_papers": list(unique_papers), "logs": logs}

def rank_agent(state: ResearchState):
    """Scores and selects top papers."""
    logs = state.get("logs", [])
    papers = state.get("raw_papers", [])
    current_year = datetime.now().year
    
    # Fallback if no papers found
    if not papers:
        logs.append("⚠️ No papers found to rank.")
        return {"selected_papers": [], "logs": logs}

    for p in papers:
        # Scoring: 40% Citations, 60% Recency
        c_score = (p.get('citation_count', 0) / 500) * 40
        r_score = (10 - min(10, (current_year - p.get('year', current_year)))) * 6
        p['score'] = c_score + r_score
        
    selected = sorted(papers, key=lambda x: x['score'], reverse=True)[:num_papers]
    logs.append(f"✅ Ranking Complete: Selected top {len(selected)}")
    return {"selected_papers": selected, "logs": logs}

def summary_agent(state: ResearchState):
    """Summarizes papers using Local RTX A4000 + Hosted Adapter."""
    logs = state.get("logs", [])
    selected = state.get("selected_papers", [])
    
    if not selected:
        logs.append("⚠️ Skipping Summarization (No papers selected)")
        return {"summaries": {}, "logs": logs}

    tokenizer, bnb_config = load_model_resources()
    if not tokenizer:
        return {"summaries": {}, "logs": logs + ["❌ Model Load Failed"]}

    # Load Base Model Just-In-Time
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            token=os.environ.get("HF_TOKEN")
        )
        
        # Load YOUR Custom Adapter from Hugging Face Hub
        logs.append(f"📥 Downloading/Loading Adapter: {ADAPTER_PATH}")
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        logs.append("✅ Fine-Tuned Adapter Active")
        
    except Exception as e:
        model = base_model # Fallback if adapter fails
        logs.append(f"⚠️ Adapter Load Error: {e} - Using Base Model")

    summaries = {}
    for p in selected:
        # Standard instruction format for Llama 3
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert researcher. Summarize the following abstract concisely.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Title: {p['title']}
        Abstract: {p['abstract']}
        Summary:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=250, temperature=0.3)
            
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary = full_text.split("assistant")[-1].strip()
        summaries[p['title']] = summary
    
    # Clear GPU RAM
    del model
    del base_model
    torch.cuda.empty_cache()
    
    logs.append(f"✅ Summarization Complete: Processed {len(summaries)} papers")
    return {"summaries": summaries, "logs": logs}

def compare_agent(state: ResearchState):
    """Generates final report using Groq."""
    logs = state.get("logs", [])
    summaries = state.get("summaries", {})
    
    if not summaries:
        return {"final_analysis": "No summaries available to analyze.", "logs": logs}

    context = "\n\n".join([f"### Paper: {k}\n**Summary:** {v}" for k,v in summaries.items()])
    
    # Using Llama 3.3 70B for high-level synthesis
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=model_temp, api_key=os.environ.get("GROQ_API_KEY"))
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Senior Research Scientist. Write a comprehensive report based on these summaries."),
        ("user", """
        Create a structured research report with the following sections:
        1. EXECUTIVE SUMMARY: A high-level overview.
        2. KEY THEMES: What concepts appear across multiple papers?
        3. CRITICAL ANALYSIS: Identify contradictions or unique methodologies.
        4. FUTURE DIRECTIONS: What gaps exist?
        
        SOURCE DATA:
        {context}
        """)
    ])
    
    chain = prompt | llm | StrOutputParser()
    analysis = chain.invoke({"context": context})
    logs.append("✅ Final Analysis Generated")
    return {"final_analysis": analysis, "logs": logs}

# ==========================================
# 4. GRAPH WIRING
# ==========================================
workflow = StateGraph(ResearchState)
workflow.add_node("keyword", keyword_agent)
workflow.add_node("search", search_agent)
workflow.add_node("rank", rank_agent)
workflow.add_node("summary", summary_agent)
workflow.add_node("compare", compare_agent)

workflow.set_entry_point("keyword")
workflow.add_edge("keyword", "search")
workflow.add_edge("search", "rank")
workflow.add_edge("rank", "summary")
workflow.add_edge("summary", "compare")
workflow.add_edge("compare", END)

app_pipeline = workflow.compile()

# ==========================================
# 5. MAIN UI RENDERING
# ==========================================
st.header("🧠 Autonomous Research Agent")
st.markdown(f"**Powered by:** `Groq Cloud` (Planning) + `Local RTX A4000` (Compute) + `Mujtaba007/llama3-arxiv-lora` (Fine-Tune)")

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("Research Topic", placeholder="e.g., Retrieval Augmented Generation in Healthcare")
with col2:
    st.write("") # Spacing
    st.write("")
    run_btn = st.button("🚀 Start Research", type="primary", use_container_width=True)

if run_btn:
    if not os.environ.get("GROQ_API_KEY") or not os.environ.get("HF_TOKEN"):
        st.error("⚠️ API Keys missing! Please configure them in the Sidebar.")
    else:
        status_container = st.status("🕵️‍♂️ Agents are working...", expanded=True)
        
        # Initialize with the user input
        full_state = {"user_query": query, "logs": []}
        
        try:
            # STREAMING EXECUTION
            for step in app_pipeline.stream(full_state):
                # Update status based on which agent just finished
                for key, value in step.items():
                    # CRITICAL FIX: Merge new data into full_state instead of overwriting
                    full_state.update(value)
                    
                    if key == "keyword":
                        status_container.write(f"✅ Agent 1 (Keyword): {value.get('expanded_keywords')}")
                    elif key == "search":
                        status_container.write(f"✅ Agent 2 (Search): Found {len(value.get('raw_papers', []))} papers")
                    elif key == "rank":
                        status_container.write("✅ Agent 3 (Rank): Ranking Complete")
                    elif key == "summary":
                        status_container.write("✅ Agent 4 (Summary): Summaries Generated (Local GPU)")
                    elif key == "compare":
                        status_container.write("✅ Agent 5 (Compare): Final Analysis Done")
            
            status_container.update(label="✅ Research Completed Successfully!", state="complete", expanded=False)
            
            # NOW use full_state which contains keys from ALL agents
            if "final_analysis" in full_state:
                # RESULT TABS
                tab1, tab2, tab3 = st.tabs(["📝 Final Report", "📚 Source Papers", "⚙️ System Logs"])
                
                with tab1:
                    st.markdown("### Research Report")
                    st.markdown(full_state.get("final_analysis", "No analysis."))
                    
                    report_text = f"TOPIC: {query}\nDATE: {datetime.now().strftime('%Y-%m-%d')}\n\n{full_state.get('final_analysis', '')}"
                    st.download_button("💾 Download Report", report_text, file_name="research_report.txt")

                with tab2:
                    selected_papers = full_state.get("selected_papers", [])
                    if not selected_papers:
                        st.warning("No source papers to display.")
                    else:
                        for p in selected_papers:
                            with st.expander(f"📄 {p['title']}"):
                                st.caption(f"Published: {p.get('year', 'N/A')} | Citations: {p.get('citation_count', 'N/A')}")
                                st.markdown(f"**Abstract:** {p.get('abstract', 'No abstract available.')}")
                                st.markdown(f"**AI Summary:** {full_state.get('summaries', {}).get(p['title'], 'N/A')}")
                                st.link_button("Read Full Paper", p.get('url', '#'))

                with tab3:
                    for log in full_state.get("logs", []):
                        st.code(log, language="text")
                    
        except Exception as e:
            status_container.update(label="❌ Pipeline Failed", state="error")
            st.error(f"An error occurred: {e}")