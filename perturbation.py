from classifier_manager import ClassifierManager
import torch

class Perturbation:
    def __init__(self, classifier_manager: ClassifierManager, target_probability: float=0.001, accuracy_threshold: float=0.9, 
                 perturbed_layers: list[int]=None, perturbated_label: str="all", repeat: int=1):
        self.classifier_manager = classifier_manager
        self.target_probability = target_probability
        self.accuracy_threshold = accuracy_threshold
        self.perturbed_layers = perturbed_layers
        self.perturbated_label = perturbated_label

        self.repeat = repeat

    def get_perturbation(self, output_hook: torch.Tensor, layer: int, eval_ppl: bool) -> torch.Tensor:
        if self.perturbed_layers is None or layer in self.perturbed_layers:
            if self.perturbated_label == "all":
                for label_name, clfr_list in self.classifier_manager.classifiers.items():
                    clfr = clfr_list[layer]
                    if not eval_ppl:
                        output_hook[0][-1,:] = self.get_token_perturbation(output_hook[0][-1,:], clfr, layer, label_name)
                    else:
                        for t in range(output_hook.shape[1]):
                            output_hook[0,t,:] = self.get_token_perturbation(output_hook[0,t,:], clfr, layer, label_name)
            else:
                clfr = self.classifier_manager.classifiers[self.perturbated_label][layer]
                for i in range(self.repeat):
                    if not eval_ppl:
                        output_hook[0][-1,:] = self.get_token_perturbation(output_hook[0][-1,:], clfr, layer, self.perturbated_label)
                    else:
                        for t in range(output_hook.shape[1]):
                            output_hook[0,t,:] = self.get_token_perturbation(output_hook[0,t,:], clfr, layer, self.perturbated_label)
        return output_hook
    
    def get_token_perturbation(self, output, clfr, layer, label_name):
        if clfr.predict_proba(output) > self.target_probability:
            perturbed_embds = self.classifier_manager.cal_perturbation(
                embds_tensor=output,
                label=label_name,
                layer=layer,
                target_prob=self.target_probability,
            )
            output += perturbed_embds[0]
        return output