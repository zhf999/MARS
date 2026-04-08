import json
import numpy as np
import os
import csv
from constants import *
from sklearn.model_selection import train_test_split
import pandas as pd

def _create_empty_dict(components_list):
    empty_dict = {}
    for comp in components_list:
        empty_dict[comp] = {0: [], 1: []}
    return empty_dict

def _restructure(texts_split, labels_split, components_list):
    split_dict = _create_empty_dict(components_list)
        
    comp_to_index = {comp: i for i, comp in enumerate(components_list)}
        
    for text, label_vector in zip(texts_split, labels_split):
        for comp_name, index in comp_to_index.items():
            
            if index < len(label_vector):
                label_value = label_vector[index]
                if label_value == 1:
                    split_dict[comp_name][1].append(text)
                else:
                    split_dict[comp_name][0].append(text)
    return split_dict
    
def prepare_jigsaw_balance(filepath="ToxicData/jigsaw_full.csv", 
                           train_ratio=0.5, 
                           total_size=400, 
                           random_state=42):
    LABEL_COLUMNS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    TEXT_COLUMN = 'comment_text'

    max_samples = total_size//2

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    df = pd.read_csv(filepath)
    
    X_train_dict = {}
    y_train_dict = {}
    X_test_dict = {}
    y_test_dict = {}
    
    np.random.seed(random_state)
    
    print(f"\nStarting to build balanced datasets for {len(LABEL_COLUMNS)} labels (positive:negative = 1:1)...")
    
    for label in LABEL_COLUMNS:
        print(f"Processing label: [{label}]")
        
    # 1. Separate positive and negative samples
    # Positive samples: rows where the label is 1
        pos_df = df[df[label] == 1]
    # Negative samples: rows where the label is 0
        neg_df = df[df[label] == 0]
        
        n_pos = len(pos_df)
        n_neg = len(neg_df)
        
        if n_pos == 0:
            print(f"  Warning: label {label} has no positive samples, skipping.")
            continue
            
    # 2. Determine number of samples
    # If max_samples is specified and there are enough positive samples, use max_samples
    # Otherwise use all positive samples
        if max_samples is not None and n_pos > max_samples:
            n_to_sample = max_samples
            # Randomly sample positive examples
            pos_df_sampled = pos_df.sample(n=n_to_sample, random_state=random_state)
        else:
            n_to_sample = n_pos
            pos_df_sampled = pos_df
            
    print(f"  - Positives: {n_pos} -> sampled: {n_to_sample}")
        
        # 3. Sample negative examples (ensure 1:1 ratio)
        # Draw the same number of negatives as positives
    if n_neg < n_to_sample:
        print(f"  Warning: not enough negatives ({n_neg} < {n_to_sample}), cannot fully balance. Using all negatives.")
        neg_df_sampled = neg_df
    else:
        neg_df_sampled = neg_df.sample(n=n_to_sample, random_state=random_state)
            
    print(f"  - Negatives: {n_neg} -> sampled: {len(neg_df_sampled)} (matching positives)")
        
    # 4. Combine and shuffle
    balanced_df = pd.concat([pos_df_sampled, neg_df_sampled])
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
    # Extract X and y
    # Note: y here is the single label for the current column (0 or 1)
    X_all = balanced_df[TEXT_COLUMN].tolist()
    y_all = balanced_df[label].values
        
    # 5. Split into train and test sets
    # stratify=y_all ensures positive/negative ratio remains 1:1 in train/test
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, 
        test_size=1-train_ratio, 
        random_state=random_state, 
        stratify=y_all
    )
        
    # 6. Store into dictionaries
    X_train_dict[label] = X_tr
    y_train_dict[label] = y_tr
    X_test_dict[label] = X_te
    y_test_dict[label] = y_te
        
    print(f"  - Done: Train ({len(X_tr)}), Test ({len(X_te)})")

    print("\nAll labels processed.")
    return X_train_dict, y_train_dict, X_test_dict, y_test_dict

def prepare_RTP(data_path="ToxicData/datasetChal.jsonl", size=100):
    texts = []

    with open(data_path,'r',encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = data.get("prompt", {}).get("text", "")
            continuation = data.get("continuation",{}).get("text", "")
            prompt_text = text
            if prompt_text is not None and prompt_text != "":
                texts.append(prompt_text)

    return texts[:size]
