
def get_local_path(model_name):
    path_dict = {
        'bert-base-uncased': 'D:/NLP/pretrained/bert-base-uncased',
        'bert-base-chinese': 'D:/NLP/pretrained/bert-base-chinese',
        'bert-base-multilingual-cased': 'D:/NLP/pretrained/bert-base-multilingual-cased',
        'hfl/chinese-bert-wwm-ext': 'D:/NLP/pretrained/hfl/chinses-bert-wwm-ext',
        'hfl/chinese-roberta-wwm-ext': 'D:/NLP/pretrained/hfl/chinese-roberta-wwm-ext',
        't5-base': 'D:/NLP/pretrained/t5-base',
        'facebook/bart-large-cnn': 'D:/NLP/pretrained/facebook/bart-large-cnn',
        'facebook/bart-base': 'D:/NLP/pretrained/facebook/bart-base',
        'sshleifer/distilbart-xsum-12-6': 'D:/NLP/pretrained/sshleifer/distilbart-cnn-12-6',
        'xlnet-base-cased': 'D:/NLP/pretrained/xlnet-base-cased',
        'sshleifer/distillmbart-12-6': 'D:/NLP/pretrained/sshleifer/distillmbart-12-6'
    }

    return path_dict[model_name]