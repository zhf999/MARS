from llm_config import cfg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

import torch.nn as nn
import torch
import numpy as np

class LayerClassifier:
    def __init__(self, llm_cfg: cfg, lr: float=0.01, max_iter: int=10000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = make_pipeline(
            PCA(n_components=64),
            LogisticRegression(solver="saga",max_iter=max_iter,class_weight="balanced")
        )

    def train(self, features: torch.tensor, labels: list, n_epoch: int=100, batch_size: int=64) -> list[float]:
        self.model.fit(features.cpu().numpy(), np.array(labels))
        return []
    
    def predict(self, tensor: torch.tensor) -> torch.tensor:
        return torch.tensor(self.model.predict(tensor.cpu().numpy()))

    def predict_proba(self, tensor: torch.tensor) -> torch.tensor:
        w, b = self.get_weights_bias()
        return torch.sigmoid(tensor @ w.T + b)
        
    def evaluate(self, X_test: torch.tensor, y_test: list) -> tuple[float, float, float]:
        predictions = self.predict(X_test) 
        true_labels = torch.tensor(y_test)
        
        pred_binary = (predictions > 0.5).float()
        correct_count = torch.sum(pred_binary == true_labels).item()
        
        y_pred_prob_np = predictions.detach().cpu().numpy()
        y_pred_bin_np = pred_binary.detach().cpu().numpy() 
        y_true_np = true_labels.detach().cpu().numpy()

        accuracy = correct_count / len(true_labels)
        
        f1 = f1_score(y_true_np, y_pred_bin_np, average='binary')
        
        try:
            auroc = roc_auc_score(y_true_np, y_pred_prob_np)
        except ValueError:
            auroc = 0.0 

        return accuracy, f1, auroc
    
    def get_weights_bias(self) -> tuple[torch.tensor, torch.tensor]:
        pca = self.model.named_steps['pca']
        lr = self.model.named_steps['logisticregression']
        w_orig_np = lr.coef_ @ pca.components_

        b_lr = lr.intercept_
        
        if pca.mean_ is not None:
            bias_correction = np.dot(w_orig_np, pca.mean_)
            b_orig_np = b_lr - bias_correction
        else:
            b_orig_np = b_lr

        return (
            torch.tensor(w_orig_np, dtype=torch.float32).to(self.device),
            torch.tensor(b_orig_np, dtype=torch.float32).to(self.device)
        )
