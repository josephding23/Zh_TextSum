from transformers import pipeline
from util.hugging import get_local_path
from transformers import MBartTokenizer, BartModel, AutoTokenizer, AutoModelWithLMHead
import random

nplcc2017sum_path = '../../models/bart-base-nplcc2017sum/best_tfmr'
unilm_path = '../../models/torch_unilm_model'
unilm_base_path = '../pretrained_thinned/microsoft/unilm-base-cased'
dataset_dir = '../../datasets/nlpcc2017textsummarization/formatted/'

article_dir = dataset_dir + 'test.source'
abstract_dir = dataset_dir + 'test.target'


with open(article_dir, 'r', encoding='UTF-8') as p:
    lines = p.readlines()
    random_num = random.randint(0, 5000)
    while len(lines[random_num]) >= 1024:
        random_num = random.randint(0, 5000)
    article = lines[random_num]

with open(abstract_dir, 'r', encoding='UTF-8') as p:
    lines = p.readlines()
    abstract = lines[random_num]

tokenizer = AutoTokenizer.from_pretrained(nplcc2017sum_path)
model = AutoModelWithLMHead.from_pretrained(nplcc2017sum_path)

model.resize_token_embeddings(len(tokenizer))

print(len(article), len(abstract))

summarizer = pipeline("summarization",
                      model=model,
                      tokenizer=tokenizer)

# print(len(ARTICLE))

# inputs = tokenizer.encode("summarize: " + ARTICLE, return_tensors="pt", max_length=512)
# outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

# print(outputs)
generated_abstract = summarizer(article, max_length=70, min_length=30)

print(f'Article: \n{article}')
print(f'Abstract: \n{abstract}')
print(f'Generated Abstract: \n{generated_abstract[0]["summary_text"] }')

