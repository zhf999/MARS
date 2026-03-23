from toxic_data import prepare_jigsaw,prepare_jigsaw_balance
from model_extraction import ModelExtraction
from classifier_manager import ClassifierManager
import pickle
import argparse

classifier_type = 'safety'

def extract_embds(model_nickname: str, train_ratio=0.3):
    prompt_train, y_train, prompt_test, y_test = prepare_jigsaw_balance(train_ratio=train_ratio)
    llm = ModelExtraction(model_nickname)
    
    X_train = llm.extract_embds(prompt_train)
    X_test = llm.extract_embds(prompt_test)
    pickle.dump(X_train, open(f"pickles/{model_nickname}_X_train.pkl", "wb"))
    pickle.dump(X_test, open(f"pickles/{model_nickname}_X_test.pkl", "wb"))

    pickle.dump(prompt_test, open(f"pickles/{model_nickname}_prompt_test.pkl", "wb"))
    pickle.dump(y_test, open(f"pickles/{model_nickname}_y_test.pkl", "wb"))

    clfr = ClassifierManager(classifier_type)
    clfr.fit(X_train, y_train, X_test, y_test)
    pickle.dump(clfr, open(f"pickles/{model_nickname}_clfr.pkl", "wb"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='mistral-7b')
    parser.add_argument('--train_ratio','-r', type=float, default=0.5)
    args = parser.parse_args()

    model_nickname = args.model
    train_ratio = args.train_ratio

    extract_embds(model_nickname, train_ratio=train_ratio)