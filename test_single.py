import pickle
import argparse
from model_generation import ModelGeneration
from perturbation import Perturbation
def generate(model_nickname, prompt, p_0=1e-6, start_layer=0, end_layer=32):
    llm_gen = ModelGeneration(model_nickname)
    llm_gen.set_perturbation(None)

    origin_completion = llm_gen.generate(prompt)['completion']

    clfr = pickle.load(open(f"pickles/{model_nickname}_clfr.pkl", "rb"))
    pert = Perturbation(clfr, target_probability=p_0, perturbed_layers=[i for i in range(start_layer, end_layer)])
    llm_gen.set_perturbation(pert)

    perturbed_completion = llm_gen.generate(prompt)['completion']

    print(f"Origin: {origin_completion}")
    print(f"Perturbed: {perturbed_completion}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='mistral-7b')
    parser.add_argument('--p0', '-p', type=float, default=1e-6)
    parser.add_argument('--prompt',type=str)
    parser.add_argument('--start_layer', '-s', type=int, default=0)
    parser.add_argument('--end_layer', '-e', type=int, default=32)
    args = parser.parse_args()

    model_nickname = args.model
    p_0 = args.p0
    prompt = args.prompt
    start_layer = args.start_layer
    end_layer = args.end_layer

    prompt = prompt

    generate(model_nickname, prompt, p_0, start_layer, end_layer)
