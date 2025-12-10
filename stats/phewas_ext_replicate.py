import os
import json
import time
import random
import requests
import threading
import pandas as pd
import concurrent.futures
from typing import List, Dict, Any, Set
from google import genai
from google.genai import types

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    # --------------------------------------
    # API SETTINGS
    # --------------------------------------
    # STRICTLY using Gemini 3.0 Pro Preview
    "model_name": "gemini-3-pro-preview", 
    
    # Throttle: Requests per minute
    "messages_per_minute": 1, 
    
    # Parallelism (throttled by the rate limiter logic below)
    "max_workers": 1, 

    # --------------------------------------
    # DATA LOGIC
    # --------------------------------------
    "significance_threshold": 0.1,  # Q_GLOBAL < 0.1

    # --------------------------------------
    # FILE PATHS & URLS
    # --------------------------------------
    "url_targets": "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/phewas%20v4%20-%20PheWeb%20TOPMED%20phenos.tsv",
    "url_results": "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/phewas_results.tsv",
    "url_metadata": "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/significant_heritability_diseases.tsv",

    "data_dir": "data",
    "file_targets": "data/targets.tsv",
    "file_results": "data/results.tsv",
    "file_metadata": "data/metadata.tsv",

    "output_tsv": "data/mappings_final.tsv",
    "output_jsonl": "data/raw_responses.jsonl", 
}

# ==========================================
# GLOBALS & LOCKS
# ==========================================
io_lock = threading.Lock()    # For atomic printing/writing
rate_lock = threading.Lock()  # For rate limiting state

# State for Rate Limiting
last_call_time = 0.0
# Calculate delay in seconds (60s / N requests)
RATE_LIMIT_DELAY = 60.0 / CONFIG["messages_per_minute"]

# Initialize Client
# Assumes GEMINI_API_KEY is in environment variables
client = genai.Client()

# ==========================================
# 1. AUTO-DOWNLOAD & SETUP
# ==========================================
def setup_environment():
    """Creates folders and downloads files if they don't exist."""
    if not os.path.exists(CONFIG["data_dir"]):
        os.makedirs(CONFIG["data_dir"])
        print(f"Created directory: {CONFIG['data_dir']}")

    files_map = [
        (CONFIG["url_targets"], CONFIG["file_targets"]),
        (CONFIG["url_results"], CONFIG["file_results"]),
        (CONFIG["url_metadata"], CONFIG["file_metadata"]),
    ]

    for url, path in files_map:
        if not os.path.exists(path):
            print(f"Downloading {url}...")
            try:
                r = requests.get(url)
                r.raise_for_status()
                with open(path, 'wb') as f:
                    f.write(r.content)
                print(f"Saved to {path}")
            except Exception as e:
                print(f"CRITICAL ERROR downloading {url}: {e}")
                exit(1)
        else:
            print(f"Found existing file: {path}")

# ==========================================
# 2. DATA PREPARATION
# ==========================================
def load_and_prep_data():
    print("--- Loading and Preprocessing Data ---")
    
    # A. Load Targets (UKBB)
    try:
        df_targets = pd.read_csv(CONFIG["file_targets"], sep='\t')
        target_strings = [
            f"{row['phenostring']} [Category: {row['category']}]" 
            for _, row in df_targets.iterrows()
        ]
    except Exception as e:
        print(f"Error loading targets: {e}")
        return [], []
    
    # B. Load Results (Source)
    try:
        df_results = pd.read_csv(CONFIG["file_results"], sep='\t')
        
        # Filter Q_GLOBAL
        df_results = df_results[df_results['Q_GLOBAL'] < CONFIG["significance_threshold"]].copy()
        
        # Clean string: Replace underscore with space
        df_results['clean_phenotype'] = df_results['Phenotype'].str.replace('_', ' ')
        
        # Deduplicate
        unique_phenos = df_results[['clean_phenotype']].drop_duplicates()
    except Exception as e:
        print(f"Error loading results: {e}")
        return [], []
    
    # C. Load Metadata
    try:
        df_meta = pd.read_csv(CONFIG["file_metadata"], sep='\t')
        
        # D. Merge
        merged = pd.merge(
            unique_phenos,
            df_meta,
            left_on='clean_phenotype',
            right_on='disease',
            how='left'
        )
        
        merged['disease_category'] = merged['disease_category'].fillna("Unknown")
        merged['icd9_codes'] = merged['icd9_codes'].fillna("None")
        merged['icd10_codes'] = merged['icd10_codes'].fillna("None")
        
        todo_list = merged.to_dict('records')
        print(f"Prepared {len(todo_list)} unique phenotypes to process.")
        return todo_list, target_strings

    except Exception as e:
        print(f"Error merging metadata: {e}")
        return [], []

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def wait_for_rate_limit():
    """
    Blocks execution until the rate limit allows a new request.
    Thread-safe.
    """
    global last_call_time
    
    with rate_lock:
        current_time = time.time()
        # Determine when we are allowed to fire next
        # It is the max of (now) OR (last_call + delay)
        next_allowed_time = max(current_time, last_call_time + RATE_LIMIT_DELAY)
        
        # Update the global last call time to this new future slot
        last_call_time = next_allowed_time
    
    # Sleep outside the lock so we don't block other threads from calculating their slots
    sleep_duration = next_allowed_time - time.time()
    if sleep_duration > 0:
        # Print only if significant sleep to avoid log spam
        if sleep_duration > 2:
            print(f"Throttling: Sleeping {sleep_duration:.2f}s...")
        time.sleep(sleep_duration)

def clean_json_string(text: str) -> str:
    """
    Robust JSON extraction. Finds the first '{' and last '}'
    to handle cases where the model wraps JSON in markdown or conversational text.
    """
    text = text.strip()
    
    # Attempt to locate the JSON object boundaries
    start = text.find('{')
    end = text.rfind('}')
    
    if start != -1 and end != -1:
        # Extract just the JSON part
        return text[start:end+1]
    
    # Fallback: just remove markdown code blocks if standard finding fails
    if text.startswith("```json"): text = text[7:]
    elif text.startswith("```"): text = text[3:]
    if text.endswith("```"): text = text[:-3]
    return text.strip()

def get_processed_phenotypes() -> Set[str]:
    processed = set()
    if not os.path.exists(CONFIG["output_jsonl"]):
        return processed
    
    print("Scanning existing output for resumability...")
    with open(CONFIG["output_jsonl"], 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                processed.add(data['input_phenotype'])
            except:
                continue
    print(f"Found {len(processed)} phenotypes already processed.")
    return processed

# ==========================================
# 4. WORKER PROCESS
# ==========================================
def process_phenotype(row: Dict, target_list: List[str]):
    source_name = row['clean_phenotype']
    
    # Shuffle targets to avoid position bias
    current_targets = target_list.copy()
    random.shuffle(current_targets)
    targets_formatted = "\n".join(current_targets)
    
    # Construct Prompt
    prompt = f"""
    TASK: Map the "Source Phenotype" (All of Us study) to "Target Phenotypes" (UK Biobank).

    ### SOURCE PHENOTYPE:
    - Name: {source_name}
    - Category: {row['disease_category']}
    - ICD-9: {row['icd9_codes']}
    - ICD-10: {row['icd10_codes']}

    ### CANDIDATE TARGET LIST (UK Biobank):
    {targets_formatted}

    ### INSTRUCTIONS:
    1. Scan the list and identify ALL items that are matches or synonyms.
    2. FROM that list of matches, choose the SINGLE BEST match.
    3. If there are no synonymous or valid matches in the list, set "has_good_match" to false.
    4. Provide reasoning in a clear paragraph.

    ### OUTPUT FORMAT (JSON ONLY):
    {{
      "all_matches": ["list", "of", "all", "matches", "found"],
      "has_good_match": true,
      "best_match": "The Exact String of the single best match",
      "reasoning": "Explanation here..."
    }}
    """

    # Enforce Rate Limit BEFORE API call
    wait_for_rate_limit()

    # --- ATOMIC PRINTING: PROMPT ---
    with io_lock:
        print(f"\n{'='*60}")
        print(f"SENDING PROMPT FOR: {source_name}")
        print(f"{'-'*60}")
        # Print prompt (truncated target list for readability)
        print(prompt.split('### CANDIDATE TARGET LIST')[0] + "... [Target List Truncated for Log] ...\n" + prompt.split('### OUTPUT FORMAT')[1]) 
        print(f"{'='*60}")

    try:
        # API Call
        response = client.models.generate_content(
            model=CONFIG["model_name"],
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        raw_text = response.text
        
        # Parse using the robust cleaner
        parsed = json.loads(clean_json_string(raw_text))
        
        # Result Object
        result_obj = {
            "input_phenotype": source_name,
            "input_icd10": row['icd10_codes'],
            "timestamp": time.time(),
            "status": "success",
            "full_prompt_sent": prompt,
            "full_response_text": raw_text,
            "parsed_response": parsed
        }

        # --- ATOMIC PRINTING & WRITING: RESPONSE ---
        with io_lock:
            print(f"\n>>> RECEIVED RESPONSE FOR: {source_name}")
            print(raw_text)
            print(f"{'='*60}\n")
            
            # 1. Append to JSONL (Raw Data)
            with open(CONFIG["output_jsonl"], 'a') as f:
                f.write(json.dumps(result_obj) + "\n")
            
            # 2. Append to TSV (Mapped Data)
            write_header = not os.path.exists(CONFIG["output_tsv"])
            
            # Safely get list
            all_matches = parsed.get('all_matches', [])
            if isinstance(all_matches, list):
                all_matches_str = "; ".join(all_matches)
            else:
                all_matches_str = str(all_matches)

            tsv_row = {
                "Source_Phenotype": source_name,
                "Has_Good_Match": parsed.get('has_good_match'),
                "Best_Match": parsed.get('best_match'),
                "All_Matches": all_matches_str,
                "Reasoning": parsed.get('reasoning'),
                "Source_ICD10": row['icd10_codes']
            }
            
            df_row = pd.DataFrame([tsv_row])
            df_row.to_csv(CONFIG["output_tsv"], sep='\t', mode='a', header=write_header, index=False)
            
            print(f"Saved {source_name} to disk.")

    except Exception as e:
        with io_lock:
            print(f"ERROR processing {source_name}: {e}")

# ==========================================
# MAIN
# ==========================================
def main():
    setup_environment()
    
    # 1. Load Data
    all_todos, target_list = load_and_prep_data()
    if not all_todos:
        print("No data to process. Exiting.")
        return

    # 2. Resumability Check
    done_ids = get_processed_phenotypes()
    remaining_todos = [x for x in all_todos if x['clean_phenotype'] not in done_ids]
    
    print(f"Total Unique Phenotypes: {len(all_todos)}")
    print(f"Already Done: {len(done_ids)}")
    print(f"Remaining: {len(remaining_todos)}")
    
    if not remaining_todos:
        print("All phenotypes processed! Exiting.")
        return

    # 3. Execution
    print(f"Starting processing with model: {CONFIG['model_name']}")
    print(f"Rate Limit: {CONFIG['messages_per_minute']} requests per minute.")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        futures = {
            executor.submit(process_phenotype, row, target_list): row['clean_phenotype']
            for row in remaining_todos
        }
        
        for future in concurrent.futures.as_completed(futures):
            pass

    print("Job Complete.")

if __name__ == "__main__":
    main()
