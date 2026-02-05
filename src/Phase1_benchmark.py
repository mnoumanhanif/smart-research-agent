import torch
import os
import json
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import google.generativeai as genai

# ==========================================
# 🔑 CONFIGURATION
# ==========================================
HF_TOKEN = "hf_"
# PASTE YOUR GEMINI KEY HERE
GEMINI_API_KEY = "AI" 

BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B"
ADAPTER_ID = "Mujtaba007/llama3-arxiv-lora"
GEMINI_MODEL = "gemini-2.5-flash"

# How many samples to test? (5 is enough for a quick benchmark, 20+ for final)
NUM_SAMPLES = 5 

# ==========================================
# 1. SETUP METRICS & JUDGE
# ==========================================
print("⚙️ Setting up Evaluators...")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
genai.configure(api_key=GEMINI_API_KEY)
judge_model = genai.GenerativeModel(GEMINI_MODEL)

def get_judge_score(original_text, summary):
    """Asks Gemini to score the summary."""
    prompt = f"""
    Rate this summary 1-5 on Fluency, Factuality, and Coverage.
    Original: {original_text[:2000]}...
    Summary: {summary}
    Return JSON: {{ "fluency": int, "factuality": int, "coverage": int }}
    """
    try:
        res = judge_model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        return json.loads(res.text)
    except:
        return {"fluency": 1, "factuality": 1, "coverage": 1}

def evaluate_pipeline(model, tokenizer, dataset, name):
    """Runs Inference -> Quant Metrics -> Qual Metrics for one model."""
    print(f"\n🚀 Evaluating: {name}...")
    preds, refs = [], []
    judge_scores = {"fluency": [], "factuality": [], "coverage": []}
    
    device = "cuda"
    
    for i, example in enumerate(tqdm(dataset)):
        # 1. Inference
        input_text = f"Summarize the following scientific article.\n\nArticle:\n{example['article']}\n\nSummary:"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=4096).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=200, temperature=0.3, top_p=0.9, 
                do_sample=True, repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )
        summary = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        
        # Filter bad gens (rejection sampling lite)
        if len(summary) < 50: continue

        preds.append(summary)
        refs.append(example['abstract'])
        
        # 2. Judge
        scores = get_judge_score(example['article'], summary)
        for k, v in scores.items():
            judge_scores[k].append(v)

    # 3. Compute Metrics
    print(f"📊 Computing Metrics for {name}...")
    r_res = rouge.compute(predictions=preds, references=refs)
    b_res = bertscore.compute(predictions=preds, references=refs, lang="en", device=device)
    
    # Averages
    avg_scores = {
        "ROUGE-1": r_res['rouge1'],
        "ROUGE-L": r_res['rougeL'],
        "BERTScore": sum(b_res['f1']) / len(b_res['f1']),
        "Fluency": sum(judge_scores['fluency']) / len(judge_scores['fluency']),
        "Factuality": sum(judge_scores['factuality']) / len(judge_scores['factuality']),
        "Coverage": sum(judge_scores['coverage']) / len(judge_scores['coverage'])
    }
    return avg_scores

# ==========================================
# 2. MAIN EXECUTION FLOW
# ==========================================
# Load Dataset
dataset = load_dataset("ccdv/arxiv-summarization", split="test").shuffle(seed=42).select(range(NUM_SAMPLES))

# --- STEP A: EVALUATE BASE MODEL ---
print("\n⏳ Loading Base Model...")
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto", token=HF_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
tokenizer.padding_side = "left"
tokenizer.pad_token_id = tokenizer.eos_token_id

base_results = evaluate_pipeline(base_model, tokenizer, dataset, "Base Llama-3")

# Free memory for next model
del base_model
torch.cuda.empty_cache()

# --- STEP B: EVALUATE LORA MODEL ---
print("\n⏳ Loading LoRA Model...")
# Reload base to attach adapter
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto", token=HF_TOKEN)
lora_model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
lora_model.eval()

lora_results = evaluate_pipeline(lora_model, tokenizer, dataset, "Fine-Tuned LoRA")

# ==========================================
# 3. FINAL COMPARISON TABLE
# ==========================================
print("\n" + "="*50)
print("🏆 FINAL PROJECT BENCHMARK RESULTS")
print("="*50)

df = pd.DataFrame([base_results, lora_results], index=["Base Model", "LoRA Model"])
# Transpose for easier reading
df_final = df.T
print(df_final)

# Save for report
df_final.to_csv("final_benchmark_results.csv")
print("\n✅ Results saved to 'final_benchmark_results.csv'")