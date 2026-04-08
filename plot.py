import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from reduction import Reduction
import pandas as pd
import torch
from torch.nn.functional import softmax
from matplotlib import pyplot as plt
from llm_config import get_cfg
import pickle
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

def create_2dlist(n_row: int, n_col: int) -> list[list]:
    return [[[] for _ in range(n_col)] for _ in range(n_row)]

def create_layer_needed(n_layer: int, step: int) -> list[int]:
    ret = [n_layer-1-i for i in range(0,n_layer,step)]
    ret.reverse()
    return ret

def plot_testacc(testacc: list[float], threshold: float):
    fig = go.Figure()
    testacc = [i * 100 for i in testacc]
    threshold = threshold * 100

    fig.add_trace(go.Scatter(
        x=list(range(len(testacc))),
        y=testacc,
        mode='lines+markers',
        name='Test Accuracy (%)'
    ))

    fig.add_trace(go.Scatter(
        x=[0, len(testacc) - 1],
        y=[threshold, threshold],
        mode='lines',
        name='Threshold'
    ))

    fig.update_layout(
        title=dict(
            text='CAV Test Accuracy of Each Layer',
            x=0.5, 
            xanchor='center' 
        ),
        xaxis_title='Layer',
        yaxis_title='Test Accuracy (%)',
        width=600,
        height=400
    )

    fig.show()

def plot_reduction(model_nickname, layers=None, char_range=30):
    X_test = pickle.load(open(f"pickles/{model_nickname}_X_test.pkl", "rb"))
    y_test = pickle.load(open(f"pickles/{model_nickname}_y_test.pkl", "rb"))
    prompt_test = pickle.load(open(f"pickles/{model_nickname}_prompt_test.pkl", "rb"))
    llm_config = get_cfg(model_nickname)

    if layers is not None:
        print(f"Plotting Layer {layers} for all labels...")
        plot_single_layer_all_labels(prompt_test, X_test, y_test, layers, char_range)
    else:
        print(f"Plotting all layers for each label...")
        for label_name, labels in y_test.items():
            plot_label_reduction(prompt_test, X_test.layers[label_name], labels, label_name, llm_config.n_layer, char_range)

def plot_single_layer_all_labels(insts: list[str], X_test, y_test, target_layer, char_range=30):
    labels_list = list(y_test.items())
    n_labels = len(labels_list)
    
    c = 3
    r = n_labels // c + 1 if n_labels % c else n_labels // c

    fig = make_subplots(
        rows=r, cols=c,
        subplot_titles=[f"({alphabet}) {label_name}" for alphabet, (label_name, _) in zip("abcdefghijk",labels_list)]
    )

    for idx, (label_name, test_labels) in enumerate(labels_list):
        row = idx // c + 1
        col = idx % c + 1

        try:
            layer_embeddings = X_test.layers[label_name][target_layer]
        except IndexError:
            print(f"Error: Layer {target_layer} out of bounds for label {label_name}")
            continue

        pca = Reduction(2)
        pca.fit(layer_embeddings)
        
        neg_test_embds = [emb for emb, label in zip(layer_embeddings, test_labels) if label == 0]
        pos_test_embds = [emb for emb, label in zip(layer_embeddings, test_labels) if label == 1]
        
        if not neg_test_embds or not pos_test_embds:
            print(f"Warning: Label {label_name} missing positive or negative samples.")
            continue

        neg_test_pca = pca.transform(neg_test_embds)
        pos_test_pca = pca.transform(pos_test_embds)

        pos_insts = [inst[:char_range] for inst, label in zip(insts, test_labels) if label == 1]
        neg_insts = [inst[:char_range] for inst, label in zip(insts, test_labels) if label == 0]

        fig.add_trace(
            go.Scatter(
                x=neg_test_pca[:, 0], y=neg_test_pca[:, 1],
                mode='markers',
                marker=dict(color='red', opacity=0.3),
                name=f'Negative',
                text=neg_insts,
                hoverinfo='text',
                showlegend=(idx == 0)
            ),
            row=row, col=col
        )

        fig.add_trace(
            go.Scatter(
                x=pos_test_pca[:, 0], y=pos_test_pca[:, 1],
                mode='markers',
                marker=dict(color='blue', opacity=0.3),
                name=f'Positive',
                text=pos_insts,
                hoverinfo='text',
                showlegend=(idx == 0)
            ),
            row=row, col=col
        )

    fig.update_layout(
        # title=dict(
        #     text=f'PCA Projection of Layer {target_layer} Across All Labels',
        #     x=0.5,
        #     xanchor='center'
        # ),
        height=r * 300,
        width=c * 300,
        showlegend=True
    )

    fig.show()

def plot_label_reduction(insts:list[str], test_embds, test_labels, label_name, n_layer, char_range=30):
    c = 8
    r = n_layer // c + 1 if n_layer % c else n_layer // c

    fig = make_subplots(
        rows=r, cols=c,
        subplot_titles=[f'Layer {i}' for i in range(n_layer)]
    )

    for i in range(r):
        for j in range(c):
            layer = i * c + j
            if layer >= n_layer:
                break
            pca = Reduction(2)

            train_data = test_embds[layer]  
            pca.fit(train_data)
            
            neg_test_embds = [ neg_embds for neg_embds,label in zip(test_embds[layer],test_labels) if label == 0]
            pos_test_embds = [ pos_embds for pos_embds,label in zip(test_embds[layer],test_labels) if label == 1]
            neg_test_pca = pca.transform(neg_test_embds)
            pos_test_pca = pca.transform(pos_test_embds)

            pos_insts = [inst[:char_range] for inst,label in zip(insts,test_labels) if label==1]
            neg_insts = [inst[:char_range] for inst,label in zip(insts,test_labels) if label==0]

            row, col = i + 1, j + 1

            fig.add_trace(
                go.Scatter(
                    x=neg_test_pca[:, 0], y=neg_test_pca[:, 1],
                    mode='markers',
                    marker=dict(color='red', opacity=0.3),
                    name='negative',
                    text=pos_insts,
                    hoverinfo='text',
                    showlegend=(i == 0 and j == 0)
                ),
                row=row, col=col
            )
            fig.add_trace(
                go.Scatter(
                    x=pos_test_pca[:, 0], y=pos_test_pca[:, 1],
                    mode='markers',
                    marker=dict(color='blue', opacity=0.3),
                    name='positive',
                    text=neg_insts,
                    hoverinfo='text',
                    showlegend=(i == 0 and j == 0)
                ),
                row=row, col=col
            )

    fig.update_layout(
        title=dict(
            text=f'PCA Projection of {label_name} Embeddings Across Layers',
            x=0.5, 
            xanchor='center' 
        ),
        height=r * 200,
        width=c * 200,
        showlegend=True
    )

    fig.show()

def plot_layer_accuracy(model_nickname , save_path=None):
    cfg = get_cfg(model_nickname)
    layers = [i for i in range(cfg.n_layer)]
    clfr = pickle.load(open(f"pickles/{model_nickname}_clfr.pkl","rb"))
    accuracies = clfr.testacc

    plt.figure(figsize=(12, 7))
    
    for label, accuracy_list in accuracies.items():
        plt.plot(layers, accuracy_list, marker='o', linestyle='-', label=str(label))

    
    plt.title(f"Accuracy per Layer for {model_nickname}", fontsize=16)
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    
    plt.ylim(0, 1.05) 
    
    plt.legend(title="Labels", loc='best') 
    
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to {save_path}")
    
    plt.show()

def plot_layer_f1(model_nickname , save_path=None):
    cfg = get_cfg(model_nickname)
    layers = [i for i in range(cfg.n_layer)]
    clfr = pickle.load(open(f"pickles/{model_nickname}_clfr.pkl","rb"))
    f1s = clfr.test_f1

    plt.figure(figsize=(12, 7))
    
    for label, accuracy_list in f1s.items():
        plt.plot(layers, accuracy_list, marker='o', linestyle='-', label=str(label))
    
    # plt.title(f"F1-score per Layer for {model_nickname}", fontsize=16)
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("F1-score", fontsize=12)
    
    plt.ylim(0, 1.05) 
    plt.legend(title="Labels", loc='best') 
    
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to {save_path}")
    
    plt.show()

def plot_layer_auroc(model_nickname , save_path=None):

    cfg = get_cfg(model_nickname)
    layers = [i for i in range(cfg.n_layer)]
    clfr = pickle.load(open(f"pickles/{model_nickname}_clfr.pkl","rb"))
    auroc = clfr.test_auroc

    plt.figure(figsize=(12, 7))
    
    for label, accuracy_list in auroc.items():
        plt.plot(layers, accuracy_list, marker='o', linestyle='-', label=str(label))
    
    plt.title(f"AUROC per Layer for {model_nickname}", fontsize=16)
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("AUROC-score", fontsize=12)

    plt.ylim(0, 1.05) 
    
    plt.legend(title="Labels", loc='best') 
    
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to {save_path}")
    
    # 总是显示图表
    plt.show()

def plot_similarity(model_nickname, save_path=None, layers=None):
 
    cfg = get_cfg(model_nickname)
    total_model_layers = cfg.n_layer
    clfr = pickle.load(open(f"pickles/{model_nickname}_clfr.pkl", "rb"))

    if layers and len(layers) > 0:
        target_layers = layers
    else:
        target_layers = list(range(total_model_layers))
    
    num_plots = len(target_layers)
    

    cols = int(np.ceil(np.sqrt(num_plots)))
    rows = int(np.ceil(num_plots / float(cols)))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5 + 2, rows * 4.5))
    
    if num_plots == 1:
        axes = np.array([axes])
    
    axes = axes.flatten()
    
    all_labels = list(clfr.classifiers.keys())
    if not all_labels:
        print("Error: 'clfr.classifiers' is empty")
        return
    
    for i, layer_idx in enumerate(target_layers):
        ax = axes[i] 
        
        if layer_idx >= total_model_layers:
            print(f"Warning: Layer {layer_idx} exceed max layer {total_model_layers}.")
            ax.set_title(f"Layer {layer_idx} (Out of Bounds)")
            ax.axis('off')
            continue

        vec_dict = {label: vecs[layer_idx].get_weights_bias()[0].cpu() for label, vecs in clfr.classifiers.items()}
        vectors = [vec_dict[label] for label in all_labels]
            
        vector_matrix = np.vstack(vectors)

        similarity_matrix = cosine_similarity(vector_matrix)
        

        sns.heatmap(similarity_matrix, 
                    ax=ax,
                    annot=True,         
                    fmt=".2f",          
                    cmap="vlag",        
                    vmin=-1, vmax=1,    
                    xticklabels=all_labels,
                    yticklabels=all_labels,
                    cbar=False,         
                    square=True)        
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for j in range(num_plots, len(axes)):
        axes[j].axis('off')
    
    sm = plt.cm.ScalarMappable(cmap="vlag", norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([]) 
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7]) 
    fig.colorbar(sm, cax=cbar_ax, label="Cosine Similarity")

    plt.tight_layout(rect=[0, 0.03, 0.92, 0.95]) 
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()