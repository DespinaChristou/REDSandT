import os
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import BertModel

ROOT_PATH = '.'


def tsv_to_dict(input_file, output_file):
    input = pd.read_csv(input_file, sep='\t', names=['id', 'description'])
    fbid_to_desc = {}
    for index, sample in input.iterrows():
        ent_fb_id = sample['id']
        ent_desc = sample['description']
        fbid_to_desc[ent_fb_id] = ent_desc

    # Save entity2FBid dictionary
    save_dict(fbid_to_desc, output_file)


def get_entity_to_FB_id(reside_train_dataset, reside_test_dataset, output_file):
    entity2FBid = {}
    # Get entities from test dataset
    print("Get entities from test set...")
    test_samples = []
    for line in open(reside_test_dataset, 'r', encoding='utf8'):
        test_samples.append(json.loads(line))
    for sample in test_samples:
        # Append {head_ent, head_FB_id} if not already in dict
        head_ent = " ".join(sample['sub'].split("_"))
        head_FB_id = sample['sub_id']
        if head_ent not in entity2FBid:
            entity2FBid[head_ent] = head_FB_id

        # Append {head_ent, head_FB_id} if not already in dict
        tail_ent = " ".join(sample['obj'].split("_"))
        tail_FB_id = sample['obj_id']
        if tail_ent not in entity2FBid:
            entity2FBid[tail_ent] = tail_FB_id
    del test_samples

    ## Get entities from TRAIN dataset
    print("Get entities from train set...")
    train_samples = []
    for line in open(reside_train_dataset, 'r', encoding='utf8'):
        train_samples.append(json.loads(line))

    for sample in train_samples:
        # Append {head_ent, head_FB_id} if not already in dict
        head_ent = " ".join(sample['sub'].split("_"))
        head_FB_id = sample['sub_id']
        if head_ent not in entity2FBid:
            entity2FBid[head_ent] = head_FB_id

        # Append {head_ent, head_FB_id} if not already in dict
        tail_ent = " ".join(sample['obj'].split("_"))
        tail_FB_id = sample['obj_id']
        if tail_ent not in entity2FBid:
            entity2FBid[tail_ent] = tail_FB_id
    del train_samples

    # Save entity2FBid dictionary
    save_dict(entity2FBid, output_file)

    return entity2FBid


def get_relative_position_ids(tokens, head_tokens, tail_tokens, max_length):
    cls_token_id = tokens.index("[CLS]")
    h_sep_token_id = tokens.index("[H-SEP]")
    t_sep_token_id = tokens.index("[T-SEP]")
    sep_token_id = tokens.index("[SEP]")
    # Get all head token ids
    head_tokens_pos = []
    for i in range(0, len(head_tokens)):
        head_token_pos = np.where(np.array(tokens) == head_tokens[i])[0]
        head_tokens_pos.extend(head_token_pos)
    head_stp_tokens = sorted(i for i in head_tokens_pos if i > t_sep_token_id)

    # Get all tail token ids
    tail_tokens_pos = []
    for i in range(0, len(tail_tokens)):
        tail_token_pos = np.where(np.array(tokens) == tail_tokens[i])[0]
        tail_tokens_pos.extend(tail_token_pos)
    tail_stp_tokens = sorted(i for i in tail_tokens_pos if i > t_sep_token_id)

    head_tail_distance = min(tail_stp_tokens) - max(head_stp_tokens)

    ## RELATIVE POSITION IDS##
    rel_pos_ids = np.zeros(max_length)
    # For CLS token, rel_pos = self.max_pos_distance
    rel_pos_ids[cls_token_id] = max_length - 1
    # Tokens in Head Tokens assigned with i=0
    rel_pos_ids[head_tokens_pos] = 0
    rel_pos_ids[cls_token_id + 1] = 0
    rel_pos_ids[h_sep_token_id] = max_length - 2
    # Tokens between head and tail
    rel_pos_ids[max(head_stp_tokens) + 1: min(tail_stp_tokens)] = np.arange(1, head_tail_distance)
    # Tail Tokens
    rel_pos_ids[h_sep_token_id + 1] = head_tail_distance
    rel_pos_ids[tail_tokens_pos] = head_tail_distance
    rel_pos_ids[t_sep_token_id] = max_length - 2
    # SEP Token
    rel_pos_ids[sep_token_id] = head_tail_distance + 1
    # Rest Tokens
    rel_pos_ids[sep_token_id:max_length] = np.arange(head_tail_distance + 1,
                                                     max_length - sep_token_id + head_tail_distance + 1)
    return rel_pos_ids


def get_rel_positions(tokens, head_tokens, tail_tokens):
    cls_token_id = tokens.index("[CLS]")
    h_sep_token_id = tokens.index("[H-SEP]")
    t_sep_token_id = tokens.index("[T-SEP]")
    sep_token_id = tokens.index("[SEP]")
    # Get all head token ids
    head_tokens_pos = []
    for i in range(0, len(head_tokens)):
        head_token_pos = np.where(np.array(tokens) == head_tokens[i])[0]
        head_tokens_pos.extend(head_token_pos)
    head_stp_tokens = sorted(i for i in head_tokens_pos if i > t_sep_token_id)

    # Get all tail token ids
    tail_tokens_pos = []
    for i in range(0, len(tail_tokens)):
        tail_token_pos = np.where(np.array(tokens) == tail_tokens[i])[0]
        tail_tokens_pos.extend(tail_token_pos)
    tail_stp_tokens = sorted(i for i in tail_tokens_pos if i > t_sep_token_id)

    head_tail_distance = min(tail_stp_tokens) - max(head_stp_tokens)

    ## HEAD RELATIVE POSITION ##
    h_rel_pos = np.zeros(len(tokens))
    # For CLS token, rel_pos = self.max_pos_distance
    h_rel_pos[cls_token_id] = 0  # max_pos_distance
    # Tokens in Head Tokens assigned with i=0
    h_rel_pos[head_tokens_pos] = 0
    h_rel_pos[cls_token_id + 1] = 0
    h_rel_pos[h_sep_token_id] = 0
    # Tokens between head and tail
    h_rel_pos[max(head_stp_tokens) + 1: min(tail_stp_tokens)] = np.arange(1, head_tail_distance)
    # Tail Tokens
    h_rel_pos[h_sep_token_id + 1] = head_tail_distance
    h_rel_pos[tail_tokens_pos] = head_tail_distance
    h_rel_pos[t_sep_token_id] = head_tail_distance
    # SEP Token
    h_rel_pos[sep_token_id] = head_tail_distance + 1

    ## TAIL RELATIVE POSITION ##
    t_rel_pos = np.zeros(len(tokens))
    # For CLS token, rel_pos = self.max_pos_distance
    t_rel_pos[cls_token_id] = 0  # max_pos_distance
    # Tokens in Tail Tokens assigned with i=0
    t_rel_pos[tail_tokens_pos] = 0
    t_rel_pos[h_sep_token_id + 1] = 0
    t_rel_pos[t_sep_token_id] = 0
    # Tokens between head and tail
    t_rel_pos[max(head_stp_tokens) + 1: min(tail_stp_tokens) + 1] = np.arange(-head_tail_distance + 1, 1)
    # Head Tokens
    t_rel_pos[cls_token_id + 1] = -head_tail_distance
    t_rel_pos[head_tokens_pos] = -head_tail_distance
    t_rel_pos[h_sep_token_id] = -head_tail_distance
    # SEP Token
    t_rel_pos[sep_token_id] = 1
    return list(h_rel_pos.astype(np.int32)), list(t_rel_pos.astype(np.int32))


def plot_train_val_loss(train_loss, valid_loss, model_name):
    output_file = os.path.join(ROOT_PATH, 'experiments/outputs', model_name, 'train_val_loss.png')

    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 0.5)  # consistent scale
    plt.xlim(0, len(train_loss) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(output_file, bbox_inches='tight')


def plot_precision_recall_curve(precision, recall, auc, model_name):
    output_file = os.path.join(ROOT_PATH, 'experiments/outputs', model_name, 'prec_rec_curve.png')

    plt.figure()
    label = model_name + ' | AUC:' + str(round(auc, 3))
    print("Label:", label)
    plt.grid(True)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.plot(recall, precision, 'r', linewidth=1, label=label)
    plt.legend(loc="upper right")
    plt.savefig(output_file)


def save_precision_recall_values(precision, recall, model_name):
    prec_rec_dict = {'prec': precision, 'rec': recall}

    # Save dict to file
    output_file = os.path.join(ROOT_PATH, 'experiments/outputs', model_name, 'prec_rec_dict.pkl')
    save_dict(prec_rec_dict, output_file)


def save_eval_metrics(p100, p200, p300, p500, p1000, p2000, pAll, auc, model_name):
    eval_metrics_dict = {'auc': auc, 'p@100': p100, 'p@200': p200, 'p@300': p300, 'p@500': p500, 'p@1000': p1000,
                         'p@2000': p2000, 'p@all': pAll}

    # Save dict to file
    output_file = os.path.join(ROOT_PATH, 'experiments/outputs', model_name, 'eval_metrics_dict.pkl')
    save_dict(eval_metrics_dict, output_file)


def save_labels_distribution_at_top_300_predictions(labels_distribution, model_name):
    # Save dict to file
    output_file = os.path.join(ROOT_PATH, 'experiments/outputs', model_name, 'labels_distribution_top300_dict.pkl')
    save_dict(labels_distribution, output_file)


def save_model_attention_weights(attentions, model_name):
    attentions_dict = {'attention': attentions}
    # Save dict to file
    output_file = os.path.join(ROOT_PATH, 'experiments/outputs', model_name, 'attention_weights.pkl')
    save_dict(attentions_dict, output_file)


def save_dict(dictionary, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)


def load_dict(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_json(sorted_pred_results, output_file):
    with open(output_file, 'w') as f:
        json.dump(sorted_pred_results, f)


def load_json(file_path):
    with open(file_path, 'rb') as f:
        return json.load(f)


def save_finetuned_model(finetuned_model, model_name):
    # Save fine-tuned model
    output_dir = os.path.join(ROOT_PATH, 'experiments/outputs', model_name, 'finetuned_model')
    print("Saving model in %s" % output_dir)
    finetuned_model.save_pretrained(output_dir)


def save_updated_tokenizer(updated_tokenizer, model_name):
    # Save finetuned model
    output_dir = os.path.join(ROOT_PATH, 'experiments/outputs', model_name, 'finetuned_model')
    print("Saving model in %s" % output_dir)
    updated_tokenizer.save_pretrained(output_dir)


def load_finetuned_model(model_dir):
    model = BertModel.from_pretrained(model_dir)
    # Copy the model to the GPU.
    # model.to(device)
    return model


def save_relational_facts(rel_facts_dict, model_name):
    rel_facts = [k for k, v in rel_facts_dict.items()]
    output_file = os.path.join(ROOT_PATH, 'experiments/outputs', model_name, 'rel_facts.json')
    save_json(rel_facts, output_file)


def save_sorted_pred_results(sorted_pred_results, model_name):
    output_file = os.path.join(ROOT_PATH, 'experiments/outputs', model_name, 'sorted_pred_results.json')
    save_json(sorted_pred_results, output_file)
