from transformers import pipeline
from pprint import pprint
from transformers import BertTokenizer, BertModel, AutoModelWithLMHead, AutoTokenizer, T5PreTrainedModel, T5Tokenizer
from util.hugging import get_local_path


name = 'xlnet-base-cased'
tokenizer = AutoTokenizer.from_pretrained(get_local_path(name))
model = AutoModelWithLMHead.from_pretrained(get_local_path(name))

text_generator = pipeline("text-generation", tokenizer=tokenizer, model=model)
print(text_generator("As far as I am concerned, I will", max_length=50, do_sample=False))
