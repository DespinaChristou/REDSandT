import argparse
import json
import os
import time

import torch

from redsandt.encoder.bert_encoder import BERTEncoder
from redsandt.framework.bag_re import BagRE
from redsandt.selector.bag_attention import BagAttention

# Pass arguments
parser = argparse.ArgumentParser(
    description='Improving Distantly-Supervised Relation Extraction through BERT-based Label & Instance Embeddings')

parser.add_argument('--train', dest="train", action='store_true', help='training mode')
parser.add_argument('--eval', dest="eval", action='store_true', help='evaluation mode')
parser.add_argument('--dataset', dest="dataset", required=True, help='dataset')
parser.add_argument('--config', dest="config", required=True, help='configuration file')
parser.add_argument('--model_dir', dest="model_dir", required=True, help='model dir')
parser.add_argument('--model_name', dest="model_name", required=True, help='model name')
args = parser.parse_args()

# Some basic settings
ROOT_PATH = '.'
DATASET = args.dataset # NYT-10 or GDS
MODEL_DIR = args.model_dir
MODEL_NAME = args.model_name
config = json.load(open(args.config))
# Create folders
if not os.path.exists('experiments/ckpt/' + DATASET + '/' + MODEL_DIR):
    os.makedirs('experiments/ckpt/' + DATASET + '/' + MODEL_DIR)
if not os.path.exists('experiments/outputs/' + DATASET + '/' + MODEL_DIR):
    os.makedirs('experiments/outputs/' + DATASET + '/' + MODEL_DIR)

ckpt = 'experiments/ckpt/' + DATASET + '/' + MODEL_DIR + '/' + MODEL_NAME + '.pth.tar'
if DATASET == 'NYT-10':
    rel2id = json.load(open(os.path.join(ROOT_PATH, 'benchmark/NYT-10-enhanced/nyt10_rel2id.json')))
elif DATASET == 'GDS':
    rel2id = json.load(open(os.path.join(ROOT_PATH, 'benchmark/GDS-enhanced/gids_rel2id.json')))

# DEFINE SENTENCE ENCODER
print('Defining the sentence encoder...')
sentence_encoder = BERTEncoder(max_length=config['encoder']['max_length'], num_labels=config['encoder']['num_labels'],
                               pretrained_model=config['encoder']['pretrained_model'],
                               drop=config['encoder']['encoder_dropout'], freeze_bert=config['encoder']['freeze_bert'],
                               text_stp=config['encoder']['text_stp'], entity_types=config['encoder'][
        'entity_types'], dataset=DATASET)

# DEFINE MODEL
print("\nDefining model...")
model = BagAttention(sentence_encoder, len(rel2id), rel2id, config['framework']['selector_dropout'])

# DEFINE TRAINING FRAMEWORK
print("\nDefining learning framework...")
model_path = DATASET + '/' + MODEL_DIR
framework = BagRE(train_path=config['train_data_path'], val_path=config['val_data_path'],
                  test_path=config['test_data_path'], model_name=model_path, model=model, ckpt=ckpt,
                  batch_size=config['framework']['batch_size'], max_epoch=config['framework']['max_epoch'],
                  lr=config['framework']['lr'], weight_decay=config['framework']['weight_decay'],
                  warmup_step_ratio=config['framework']['warmup_step_ratio'], opt=config['framework']['opt'],
                  weighted_loss=config['framework']['weighted_loss'], bag_size=config['framework']['bag_size'])

# TRAIN MODEL
if args.train:
    print("\nTraining model...")
    start = time.time()
    framework.train_model()
    end = time.time()
    print("Training time: ", end - start, "sec.")

# EVALUATE MODEL
if args.eval:
    print("\nEvaluate model on testing data...")
    start = time.time()
    framework.load_state_dict(torch.load(ckpt)['state_dict'])
    result = framework.eval_model(framework.test_loader, save_eval_metrics=True)
    end = time.time()
    print("Testing time: ", end - start, "sec.")

    # Print Statistics
    print('AUC: {}'.format(result['auc']))
    print('P@100: {}'.format(result['p@100']))
    print('P@200: {}'.format(result['p@200']))
    print('P@300: {}'.format(result['p@300']))
    print('P@500: {}'.format(result['p@500']))
    print('P@1000: {}'.format(result['p@1000']))
    print('P@2000: {}'.format(result['p@2000']))
    print('P@all: {}'.format(result['p@all']))
    print('\nRelation Distribution on Top 300 predictions:')
    for key, value in result['rel_dist_at_300'].items():
        print(key, ": ", value)
