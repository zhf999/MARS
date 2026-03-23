from utils import evaluate_model_ppl
from model_generation import ModelGeneration
import pickle
from perturbation import Perturbation

if __name__ == "__main__":
    model_nickname = "llama3-8b"
    p_0 = 1
    llm_gen = ModelGeneration(model_nickname)
    llm_gen.eval_ppl = True
    ppl_results = []
    
    clfr = pickle.load(open(f"pickles/{model_nickname}_clfr.pkl", "rb"))
    pert = Perturbation(clfr, target_probability=p_0, perturbed_layers=[i for i in range(15, 32)])
    llm_gen.set_perturbation(pert)

    model_ppl = evaluate_model_ppl(llm_gen.model, llm_gen.tokenizer)
    print(f"Model {model_nickname} PPL on WikiText-2: {model_ppl}")
    ppl_results.append(model_ppl)

    pickle.dump("ppl_results.pkl", ppl_results)