
def get_local_path(model_name):
    path_dict = {
        'bert-base-uncased': 'D:/NLP/bert-base-uncased',
        'bert-base-chinese': 'D:/NLP/bert-base-chinese',
        'bert-base-multilingual-cased': 'D:/NLP/bert-base-multilingual-cased',
        'hfl/chinese-bert-wwm-ext': 'D:/NLP/hfl/chinses-bert-wwm-ext',
        'hfl/chinese-roberta-wwm-ext': 'D:/NLP/hfl/chinese-roberta-wwm-ext',
        't5-base': 'D:/NLP/t5-base',
        'facebook/bart_from_transformers-large-cnn': 'D:/NLP/facebook/bart_from_transformers-large-cnn',
        'sshleifer/distilbart-xsum-12-6': 'D:/NLP/sshleifer/distilbart-cnn-12-6',
        'xlnet-base-cased': 'D:/NLP/xlnet-base-cased'
    }

    return path_dict[model_name]