import requests
import time
from tqdm.auto import tqdm
import torch
from datasets import load_dataset
import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import functional as F
from tqdm.auto import tqdm
from constants import *

def compute_text_perplexity(completions):
    if completions is None or len(completions) == 0:
        return [], float('nan')

    model_name = 'gpt2'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE_DIR, token=HF_TOKEN).to(device)
    model.eval()

    batch_size = 8
    ppls = []
    with torch.no_grad():
        for i in tqdm(range(0, len(completions), batch_size)):
            batch_texts = completions[i:i+batch_size]
            enc = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True)
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc.get('attention_mask', torch.ones_like(input_ids)).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()

            B, Tm, V = shift_logits.shape

            loss_flat = F.cross_entropy(shift_logits.view(-1, V), shift_labels.view(-1), reduction='none')
            loss = loss_flat.view(B, Tm)

            token_counts = shift_mask.sum(dim=1).float()
            token_counts_clamped = token_counts.clone()
            token_counts_clamped[token_counts_clamped == 0] = 1.0

            per_sample_nll = (loss * shift_mask).sum(dim=1) / token_counts_clamped
            per_sample_ppl = torch.exp(per_sample_nll)

            per_sample_ppl = per_sample_ppl.cpu().tolist()
            for idx_in_batch, cnt in enumerate(token_counts.cpu().tolist()):
                if cnt == 0:
                    ppls.append(float('nan'))
                else:
                    ppls.append(float(per_sample_ppl[idx_in_batch]))
    import numpy as _np
    valid = [x for x in ppls if not (_np.isnan(x) or _np.isinf(x))]
    mean_ppl = float(_np.mean(valid)) if len(valid) > 0 else float('nan')
    return ppls, mean_ppl

def evaluate_model_ppl(
    model,
    tokenizer,
    dataset_subset: str = "wikitext-2-raw-v1", 
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    stride: int = 512
):
    print(f"Loading dataset: {dataset_subset}...")
    test_data = load_dataset("wikitext", dataset_subset, split="test")
    
    encodings = tokenizer("\n\n".join(test_data["text"]), return_tensors="pt")
    
    max_length = getattr(model.config, "n_positions", 1024)
    
    if stride > max_length:
        stride = max_length

    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0

    print("Computing Model PPL...")

    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        
        target_ids = input_ids.clone()
        
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        
        if end_loc == seq_len:
            break

    total_nll = torch.stack(nlls).sum()
    ppl = torch.exp(total_nll / end_loc)
    
    result = ppl.item()
    print(f"PPL on {dataset_subset}: {result:.2f}")
    
    return result


def perspective_api_request(text, max_retries=5, retry_delay=3):
    payload = {
        "comment": {"text": text},
        "requestedAttributes": {
            "TOXICITY":{}
        }
    }

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(PERSPECTIVE_URL, json=payload, proxies=PROXIES, timeout=15)
            r.raise_for_status()
            data = r.json()["attributeScores"]
            return data["TOXICITY"]["summaryScore"]["value"]
        except Exception as e:
            print(f"Request failed for {attempt} times: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(e.response.text)
            if attempt < max_retries:
                print(f"Retry after {retry_delay} seconds.")
                time.sleep(retry_delay)
            else:
                continue

def evaluate_toxicity(texts):
    scores = []
    for text in tqdm(texts):
        scores.append(perspective_api_request(text))
    
    avg_scores = sum(scores)/len(scores)
    return scores, avg_scores