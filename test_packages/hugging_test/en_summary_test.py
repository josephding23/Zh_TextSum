from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from util.hugging import get_local_path

# see ``examples/summarization/bart_from_transformers/run_eval.py`` for a longer example
model = BartForConditionalGeneration.from_pretrained(get_local_path('facebook/bart-base'))
tokenizer = BartTokenizer.from_pretrained(get_local_path('facebook/bart-base'))

ARTICLE_TO_SUMMARIZE = '''
Gliding past President Donald Trump's sexist depictions of her as "mean" and "nasty," the senator from California shredded Trump's White House record with the agility that comes from her years as a courtroom prosecutor. Yet she delivered those critiques with bright notes of hope and optimism -- accentuated by the smiles that are expected from female politicians.
"The President's mismanagement of the pandemic has plunged us into the worst economic crisis since the Great Depression, and we're experiencing a moral reckoning with racism and systemic injustice that has brought a new coalition of conscience to the streets of our country demanding change," Harris said at the afternoon event in Wilmington, Delaware.
'''
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')

# Generate Summary
summary_ids = model.generate(inputs['input_ids'], num_beams=10, max_length=15, early_stopping=True)
print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])