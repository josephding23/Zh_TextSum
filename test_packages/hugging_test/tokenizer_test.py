from transformers import pipeline
from transformers import AutoModelWithLMHead, AutoTokenizer


path_dict = {
    'bert-base-uncased': 'D:/NLP/bert-base-uncased'
}


tokenizer = AutoTokenizer.from_pretrained(path_dict['bert-base-uncased'])
model = AutoModelWithLMHead.from_pretrained(path_dict['bert-base-uncased'])

print(tokenizer.tokenize("I have a new GPU!"))
