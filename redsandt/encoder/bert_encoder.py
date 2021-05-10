import torch
import torch.nn as nn
import numpy as np
import utils
from transformers import BertModel, BertTokenizer


class BERTEncoder(nn.Module):
    def __init__(self, max_length, num_labels, pretrained_model, drop=0.1, freeze_bert=False, text_stp=True,
                 entity_types=False, dataset='NYT-10'):
        """
        Args:
            max_length: max length of sentence
            num_labels: number of classification (relation) labels
            pretrained_model: specific pretrained model
            drop: dropout ration. Defaults to 0.1
            freeze_bert: Whether to freeze or finetune model weights
            text_stp: Whether to use STP version of input sentence
            entity_types: Whether to use entity types as additional information
        """
        super().__init__()
        # Instantiating BERT model object and tokenizer
        self.model = BertModel.from_pretrained(pretrained_model)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)

        # Define parameters
        self.max_length = max_length
        self.num_labels = num_labels
        self.text_stp = True if text_stp=="True" else False
        self.drop = nn.Dropout(p=drop)
        freeze_bert = True if freeze_bert=="True" else False
        entity_types = True if entity_types=="True" else False

        # Initialize layers
        self.softmax = nn.Softmax(-1)
        self.encoder_hidden_size = self.model.config.hidden_size
        self.hidden_size = self.encoder_hidden_size * 2
        self.rel_alias_layer = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size)
        torch.nn.init.xavier_uniform_(self.rel_alias_layer.weight)
        torch.nn.init.normal_(self.rel_alias_layer.bias)

        self.head_description_layer = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size)
        torch.nn.init.xavier_uniform_(self.head_description_layer.weight)
        torch.nn.init.normal_(self.head_description_layer.bias)

        self.tail_description_layer = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size)
        torch.nn.init.xavier_uniform_(self.tail_description_layer.weight)
        torch.nn.init.normal_(self.tail_description_layer.bias)

        self.linear_layer = nn.Linear(self.hidden_size, self.hidden_size)
        torch.nn.init.xavier_uniform_(self.linear_layer.weight)
        torch.nn.init.normal_(self.linear_layer.bias)

        # Freeze specified bert layers if selected
        if freeze_bert:
            freeze_layers = "0,1,2,3,4,5,6,7"
            if freeze_layers is not "":
                print("\nFREEZING BERT WEIGHTS...")
                layer_indexes = [int(x) for x in freeze_layers.split(",")]
                for layer_idx in layer_indexes:
                    for param in list(self.model.encoder.layer[layer_idx].parameters()):
                        param.requires_grad = False
                    print("Froze Layer: ", layer_idx)

        # Add special tokens for entity types if mask_entity is activated
        if entity_types:
            # Read special tokens
            if dataset=='NYT-10':
                ENT_SPECIAL_TOKENS_FILE = "benchmark/NYT-10-enhanced/entity_types_list.pkl"
            elif dataset == 'GDS':
                ENT_SPECIAL_TOKENS_FILE = "benchmark/GDS-enhanced/entity_types_list.pkl"
            ent_special_tokens = utils.load_dict(ENT_SPECIAL_TOKENS_FILE)
            special_tokens = list(ent_special_tokens)
            special_h_sep_token = list(['[H-SEP]'])
            special_t_sep_token = list(['[T-SEP]'])
            self.tokenizer.add_tokens(special_tokens)
            self.tokenizer.add_tokens(special_h_sep_token)
            self.tokenizer.add_tokens(special_t_sep_token)
            print('We have added', special_tokens + special_h_sep_token + special_t_sep_token, 'entity tokens')
            print("Resizing token embeddings in our model.")
            self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, token, att_mask):
        """
        Args:
            token: (B, L), token_ids including indexes of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representation for sentences
        """
        # 1. Get model' s sentence embeddings
        # Take the contextualized word embeddings (hidden representations) of each token (B, L)
        hidden, cls_hidden = self.model(token, attention_mask=att_mask)  # (Batch*Bag, L, H]

        # 1. Get positional embeddings
        head_sent_positions = torch.zeros((token.size()[0], self.max_length)).long().cuda()
        tail_sent_positions = torch.zeros((token.size()[0], self.max_length)).long().cuda()

        # Get special token ids
        h_sep_token_id = self.tokenizer.encode(['[H-SEP]'], add_special_tokens=False)[0]
        t_sep_token_id = self.tokenizer.encode(['[T-SEP]'], add_special_tokens=False)[0]
        for i, tokens in enumerate(token):
            h_sep_token = (tokens == h_sep_token_id).nonzero()
            t_sep_token = (tokens == t_sep_token_id).nonzero()

            # [CLS] h_type h [H-SEP] t_type t [T-SEP] stp tokens [SEP]
            # Find positions of head in spt tokens
            head_tokens = tokens[2: h_sep_token]
            head_tokens_all_pos = torch.tensor(np.array([])).long().cuda()
            for j in range(0, len(head_tokens)):
                head_token_pos = (tokens == head_tokens[j]).nonzero()
                head_tokens_all_pos = torch.cat((head_tokens_all_pos, head_token_pos), 0)
            head_tokens_all_pos = head_tokens_all_pos[head_tokens_all_pos > t_sep_token]

            # Find positions of tail in sentence encoding
            tail_tokens = tokens[h_sep_token + 2: t_sep_token]
            tail_tokens_all_pos = torch.tensor(np.array([])).long().cuda()
            for j in range(0, len(tail_tokens)):
                tail_token_pos = (tokens == tail_tokens[j]).nonzero()
                tail_tokens_all_pos = torch.cat((tail_tokens_all_pos, tail_token_pos), 0)
            tail_tokens_all_pos = tail_tokens_all_pos[tail_tokens_all_pos > t_sep_token]

            # Attention on Head, Tail of sentence vectors
            head_sent_positions[i][head_tokens_all_pos] = 1
            tail_sent_positions[i][tail_tokens_all_pos] = 1

        # Get head, tail hidden representation
        head_hidden = (head_sent_positions.unsqueeze(2) * hidden).sum(1)
        tail_hidden = (tail_sent_positions.unsqueeze(2) * hidden).sum(1)

        # Get rel_alias representation (t-h)
        rel_alias = torch.tanh(self.rel_alias_layer(tail_hidden.sub(head_hidden)))

        # Get rel_alias attention on cls_hidden vector
        att_score = (hidden * rel_alias.unsqueeze(1)).sum(-1)
        softmax_att_score = self.softmax(att_score)
        hidden_weighted_rep = (softmax_att_score.unsqueeze(-1) * hidden).sum(1)

        # Final Sentence Representation
        x = torch.cat([rel_alias, hidden_weighted_rep], 1)  # (B, 2H)

        # Combine (linearly) model's entity embeddings
        x = self.linear_layer(x)  # (B, 2H)

        # Add dropout
        x = self.drop(x)
        return x

    def tokenize(self, item):
        """
        Args:
            item: data instance containing 'text' / 'stp', 'h_word' and 't_word', 'h_ne', 't_ne'
        Return:
            input_ids, att_mask
        """
        # Get generic info
        head = item['h_word']
        tail = item['t_word']
        head_type = item['h_ne']
        tail_type = item['t_ne']

        if self.text_stp:
            # 1. Get sentence stp text
            sentence = item['stp']
        else:
            # 1. Get sentence full text
            sentence = item['text']

        # Tokenize input tokens
        head_tokens = self.tokenizer.tokenize(head)
        tail_tokens = self.tokenizer.tokenize(tail)
        sent_tokens = self.tokenizer.tokenize(sentence)

        # Get input representation
        tokens = ['[CLS]']
        # Head Information
        tokens.append(head_type)
        for i, token in enumerate(head_tokens):
            tokens.append(token)
        tokens.append('[H-SEP]')
        # Tail Information
        tokens.append(tail_type)
        for i, token in enumerate(tail_tokens):
            tokens.append(token)
        tokens.append('[T-SEP]')
        # Sentence Tokens
        for i, token in enumerate(sent_tokens):
            tokens.append(token)

        # Clip sentence if longer to given max_length
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length - 1 - len(tail_tokens)]
            for i, token in enumerate(tail_tokens):
                tokens.append(token)
        tokens.append('[SEP]')

        # PREPARE INPUT
        # 1a. Get Indexed tokens
        inputs = self.tokenizer.encode_plus(tokens, truncation=True, max_length=self.max_length, pad_to_max_length=True,
                                            add_special_tokens=False)
        # 1b. Pad input_ids to max_length
        input_ids = torch.LongTensor(inputs['input_ids']).unsqueeze(0).unsqueeze(0)
        # 2. Get Attention mask
        att_mask = torch.FloatTensor(inputs['attention_mask']).unsqueeze(0).unsqueeze(0)

        return input_ids, att_mask
