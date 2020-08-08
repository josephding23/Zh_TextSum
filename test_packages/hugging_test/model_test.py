from transformers import pipeline
from util.hugging import get_local_path

unmasker = pipeline('fill-mask', model=get_local_path('bert-base-uncased'), tokenizer=get_local_path('bert-base-uncased'))
print(unmasker("The man worked as a [MASK]."))