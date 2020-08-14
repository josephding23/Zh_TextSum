# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

# the proper usage is documented in the README, you need to specify data_dir, output_dir and model_name_or_path
# run ./finetune.sh --help to see all the possible options
python finetune.py \
    --data_dir=.'D:/NLP/Datasets/nlpcc2017textsummarization/formatted' \
    --output_dir='D:/NLP/models/bart-base-nplcc2017sum' \
    --model_name_or_path= 'D:/NLP/pretrained/facebook/bart-base' \
    --learning_rate=3e-5 \
    --gpus 1  \
    --do_train \
    --do_predict \
    --n_val 1000 \
    --val_check_interval 0.1 \
    --train_batch_size 8 \
    --eval_batch_size 8