from transformers import pipeline
from pprint import pprint
from transformers import BertTokenizer, BertModel, AutoModelWithLMHead, AutoTokenizer, T5PreTrainedModel, T5Tokenizer
from util.hugging import get_local_path


name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(get_local_path(name))
model = AutoModelWithLMHead.from_pretrained(get_local_path(name))

nlp = pipeline("fill-mask", tokenizer=tokenizer, model=model)
pprint(nlp(f"I am a {nlp.tokenizer.mask_token} who licks pussy."))
