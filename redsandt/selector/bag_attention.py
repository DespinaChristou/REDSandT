import torch
from torch import nn


class BagAttention(nn.Module):
    """
    Instance attention for bag-level relation extraction.
    """

    def __init__(self, sentence_encoder, num_class, rel2id, drop):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
            drop: dropout probability
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.fc = nn.Linear(self.sentence_encoder.hidden_size, num_class)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.normal_(self.fc.bias)

        self.softmax = nn.Softmax(-1)
        self.drop = nn.Dropout(p=drop)
        self.rel2id = rel2id
        self.id2rel = {}
        for rel, id in rel2id.items():
            self.id2rel[id] = rel

    def infer(self, bag):
        """
        Args:
            bag: bag of sentences with the same entity pair
        Return:
            (relation, score)
        """

        self.eval()
        tokens = []
        masks = []
        for item in bag:
            if 'stp' in item:
                token, mask = self.tokenizer(item['stp'])
            else:
                token, mask = self.tokenizer(item['text'])
            tokens.append(token)
            masks.append(mask)
        tokens = torch.cat(tokens, 0)
        masks = torch.cat(masks, 0)
        scope = torch.tensor([[0, len(bag)]]).long()
        bag_logits = self.forward(None, scope, tokens, masks, train=False).squeeze(0)
        score, pred = bag_logits.max()
        score = score.item()
        pred = pred.item()
        rel = self.id2rel[pred]
        return (rel, score)

    def forward(self, label, scope, token, mask, train=True,
                bag_size=0):
        """
        Forward pass of Bag. Useful notation:
        - B: batch_size
        - bag: bag_size
        - nsum: no. of samples in each bag
        - L: max_length of sentence embeddings (64)
        - H: hidden size (can be a multiple of 768)
        - N: no. of labels (53 in NYT dataset)
        Args:
            label: (B), label of the bag
            scope: (B), scope for each bag
            token: (nsum, L), index of tokens
            mask: (nsum, L)
        Return:
            logits, (B, N)
        """
        if bag_size > 0:
            token = token.view(-1, token.size(-1))
            mask = mask.view(-1, mask.size(-1))
        else:
            begin, end = scope[0][0], scope[-1][1]
            token = token[:, begin:end, :].view(-1, token.size(-1))
            if mask is not None:
                mask = mask[:, begin:end, :].view(-1, mask.size(-1))
            scope = torch.sub(scope, torch.zeros_like(scope).fill_(begin))

        # Get sentence embeddings (B x bag, n x H)
        rep = self.sentence_encoder(token, mask)

        # Attention
        # attention in train logits
        if train:
            if bag_size == 0:
                bag_rep = []
                query = torch.zeros((rep.size(0))).long()
                if torch.cuda.is_available():
                    query = query.cuda()
                for i in range(len(scope)):
                    query[scope[i][0]:scope[i][1]] = label[i]
                att_mat = self.fc.weight.data[query]  # (nsum, H)
                att_score = (rep * att_mat).sum(-1)  # (nsum)

                for i in range(len(scope)):
                    bag_mat = rep[scope[i][0]:scope[i][1]]  # (n, H)
                    softmax_att_score = self.softmax(att_score[scope[i][0]:scope[i][1]])  # (n)
                    bag_rep.append(
                        (softmax_att_score.unsqueeze(-1) * bag_mat).sum(0))  # (n, 1) * (n, H) -> (n, H) -> (H)
                bag_rep = torch.stack(bag_rep, 0)  # (B, H)
            else:
                batch_size = label.size(0)
                # Get queries: label for each bag in the batch (B, 1)
                query = label.unsqueeze(1)

                # Get attention matrix (B, 1, 2H) Attention based on query label (self.fc.weight.data.shape = (53, H))
                att_mat = self.fc.weight.data[query]

                # Get representation for bags in batch (B, bag, H)
                rep = rep.view(batch_size, bag_size, -1)

                # Get attention score for each bag (B, bag)
                att_score = (rep * att_mat).sum(-1)

                # Normalize bag's attention scores (B, bag)
                softmax_att_score = self.softmax(att_score)  # (B, bag)

                # Get bag representation (B, bag, 1) * (B, bag, H) -> (B, bag, H) -> (B, H)
                bag_rep = (softmax_att_score.unsqueeze(-1) * rep).sum(1)

            # Apply dropout
            bag_rep = self.drop(bag_rep)

            # Normalize bag representation (B, N)
            bag_logits = self.fc(bag_rep)

        # attention in test logits
        else:
            if bag_size == 0:
                bag_logits = []
                # Get attention scores (nsum, H) * (H, N) -> (nsum, N)
                att_score = torch.matmul(rep, self.fc.weight.data.transpose(0, 1))
                for i in range(len(scope)):
                    # Get bag matrix (n, H)
                    bag_mat = rep[scope[i][0]:scope[i][1]]

                    # Normalize attention scores (N, (softmax)n)
                    softmax_att_score = self.softmax(att_score[scope[i][0]:scope[i][1]].transpose(0, 1))

                    # Get representation for each relation (N, n) * (n, H) -> (N, H)
                    rep_for_each_rel = torch.matmul(softmax_att_score, bag_mat)

                    # Normalize representation # ((each rel)N, (logit)N)
                    logit_for_each_rel = self.softmax(self.fc(rep_for_each_rel))
                    logit_for_each_rel = logit_for_each_rel.diag()  # (N)
                    bag_logits.append(logit_for_each_rel)
                bag_logits = torch.stack(bag_logits, 0)
            else:
                # Get batch size
                batch_size = rep.size(0) // bag_size

                # Get attention score (nsum, H) * (H, N) -> (nsum, N)
                att_score = torch.matmul(rep, self.fc.weight.data.transpose(0, 1))

                # Get attention scores for each Batch, bag (B, bag, N)
                att_score = att_score.view(batch_size, bag_size, -1)
                # Normalize attention scores (B, N, (softmax)bag)
                softmax_att_score = self.softmax(att_score.transpose(1, 2))

                # Get representation for each Batch, bag
                rep = rep.view(batch_size, bag_size, -1)  # (B, bag, H)

                # Get representation for each bag (B, N, bag) * (B, bag, H) -> (B, N, H)
                rep_for_each_rel = torch.matmul(softmax_att_score, rep)

                # Normalize bag representation (B, N)
                bag_logits = self.softmax(self.fc(rep_for_each_rel)).diagonal(dim1=1, dim2=2)

        return bag_logits
