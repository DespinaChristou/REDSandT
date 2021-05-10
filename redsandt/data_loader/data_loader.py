import torch
import random
import torch.utils.data as data
import numpy as np
import pandas as pd
from collections import Counter
import sklearn.metrics
import utils

np.seterr(divide='ignore', invalid='ignore')


class BagREDataset(data.Dataset):
    """
    Bag-level relation extraction dataset. Note that relation of NA should be named as 'NA'.
    """

    def __init__(self, path, rel2id, tokenizer, entpair_as_bag=False, bag_size=0, mode=None):
        """
        Args:
            path: path of the input file
            rel2id: dictionary of relation->id mapping
            tokenizer: function of tokenizing
            entpair_as_bag: if True, bags are constructed based on same
                entity pairs instead of same relation facts (ignoring 
                relation labels)
            bag_size: bag size
            mode: training model. Defaults to multi-instance (bag) training
        """

        super().__init__()
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.entpair_as_bag = entpair_as_bag
        self.bag_size = bag_size

        if "NYT-10" in path:
            self.data = pd.read_json(path, encoding='utf8')
            self.data = self.data.to_dict('records')

            # Construct bag-level dataset
            if mode == None:
                self.weight = np.zeros((len(self.rel2id)), dtype=np.float32)
                self.bag_scope = []
                self.name2id = {}
                self.bag_name = []
                self.facts = {}
                for idx, item in enumerate(self.data):
                    rel_fact = (item['h_id'], item['t_id'], item['relation'])
                    if item['relation'] != 'NA':
                        self.facts[rel_fact] = 1
                    if entpair_as_bag:
                        name = (item['h_id'], item['t_id'])
                    else:
                        name = rel_fact
                    if name not in self.name2id:
                        self.name2id[name] = len(self.name2id)
                        self.bag_scope.append([])
                        self.bag_name.append(name)
                    self.bag_scope[self.name2id[name]].append(idx)
                    self.weight[self.rel2id[item['relation']]] += 1.0
                self.weight = np.float32(1.0 / (self.weight ** 0.05))
                self.weight = torch.from_numpy(self.weight)
            else:
                pass
        elif "GDS" in path:
            self.data = pd.read_csv(path, sep='\t', encoding='utf-8')
            self.data = self.data.to_dict('records')

            # Construct bag-level dataset
            if mode == None:
                self.weight = np.zeros((len(self.rel2id)), dtype=np.float32)
                self.bag_scope = []
                self.name2id = {}
                self.bag_name = []
                self.facts = {}
                for idx, item in enumerate(self.data):
                    rel_fact = (item['h_FB_ID'], item['t_FB_ID'], item['relation'])
                    if item['relation'] != "no_relation": #
                        self.facts[rel_fact] = 1
                    if entpair_as_bag:
                        name = (item['h_FB_ID'], item['t_FB_ID'])
                    else:
                        name = rel_fact
                    if name not in self.name2id:
                        self.name2id[name] = len(self.name2id)
                        self.bag_scope.append([])
                        self.bag_name.append(name)
                    self.bag_scope[self.name2id[name]].append(idx)
                    self.weight[self.rel2id[item['relation']]] += 1.0
                self.weight = np.float32(1.0 / (self.weight ** 0.05))
                self.weight = torch.from_numpy(self.weight)
            else:
                pass


    def __len__(self):
        return len(self.bag_scope)

    def __getitem__(self, index):
        bag = self.bag_scope[index]
        if self.bag_size > 0:
            if self.bag_size <= len(bag):
                resize_bag = random.sample(bag, self.bag_size)
            else:
                resize_bag = bag + list(np.random.choice(bag, self.bag_size - len(bag)))
            bag = resize_bag

        seqs = None
        rel = self.rel2id[self.data[bag[0]]['relation']]
        for sent_id in bag:
            item = self.data[sent_id]
            seq = list(self.tokenizer(item))
            if seqs is None:
                seqs = []
                for i in range(len(seq)):
                    seqs.append([])
            for i in range(len(seq)):
                seqs[i].append(seq[i])
        for i in range(len(seqs)):
            seqs[i] = torch.cat(seqs[i], 0)  # (bag_size, L)

        return [rel, self.bag_name[index], len(bag)] + seqs

    def collate_fn(data):
        data = list(zip(*data))
        label, bag_name, count = data[:3]
        seqs = data[3:]
        for i in range(len(seqs)):
            seqs[i] = torch.cat(seqs[i], 0)  # (sumn, L)
            seqs[i] = seqs[i].expand((torch.cuda.device_count(),) + seqs[i].size())
        scope = []
        start = 0
        for c in count:
            scope.append((start, start + c))
            start += c
        assert (start == seqs[0].size(1))
        scope = torch.tensor(scope).long()
        label = torch.tensor(label).long()
        return [label, bag_name, scope] + seqs

    def collate_bag_size_fn(data):
        data = list(zip(*data))
        label, bag_name, count = data[:3]
        seqs = data[3:]
        for i in range(len(seqs)):
            seqs[i] = torch.stack(seqs[i], 0)
        scope = []
        start = 0
        for c in count:
            scope.append((start, start + c))
            start += c
        label = torch.tensor(label).long()
        return [label, bag_name, scope] + seqs

    def eval(self, pred_result, model_name, save_eval_metrics=False):
        """
        Args:
            pred_result: a list with dict {'entpair': (head_id, tail_id), 'relation': rel, 'score': score}.
                Note that relation of NA should be excluded.
            model_name: name of the model
            save_eval_metrics: declares whether to store evaluation metrics or not
        Return:
            {'prec': narray[...], 'rec': narray[...], 'auc': xx, 'p@all': xx, 'p@100': xx, 'p@200': xx, 'p@300': xx,
             'p@500': xx, 'p@1000': xx, 'p@2000': xx, 'rel_dist_at_300': dict{...}, 'rel_facts': narray[...],
             'sorted_pred_results': narray[...], 'rel_pos_dist_at_300':dict{...}}
                prec (precision) and rec (recall) are in micro style.
                prec (precision) and rec (recall) are sorted in the decreasing order of the score.
        """
        sorted_pred_result = sorted(pred_result, key=lambda x: x['score'], reverse=True)
        prec = []
        rec = []
        correct = 0
        total = len(self.facts)
        for i, item in enumerate(sorted_pred_result):
            if (item['entpair'][0], item['entpair'][1], item['relation']) in self.facts:
                correct += 1
            prec.append(float(correct) / float(i + 1))
            rec.append(float(correct) / float(total))
        auc = np.around(sklearn.metrics.auc(x=rec, y=prec), 4)
        np_prec = np.array(prec)
        np_rec = np.array(rec)

        def prec_at_n(n):
            correct = 0
            for i, item in enumerate(sorted_pred_result[:n]):
                if (item['entpair'][0], item['entpair'][1], item['relation']) in self.facts:
                    correct += 1
            return (correct / n)

        prec_at_all = prec_at_n(len(sorted_pred_result))
        prec_at_100 = prec_at_n(100)
        prec_at_200 = prec_at_n(200)
        prec_at_300 = prec_at_n(300)
        prec_at_500 = prec_at_n(500)
        prec_at_1000 = prec_at_n(1000)
        prec_at_2000 = prec_at_n(2000)
        rel_at_300 = [x['relation'] for x in sorted_pred_result[0:300]]
        rel_dist_at_300 = dict(Counter(rel_at_300))
        rel_pos_dist_at_300 = dict(Counter([x['relation'] for x in sorted_pred_result[0:300] if x['score'] > 0.5]))

        # Return the eval metrics
        if save_eval_metrics:
            print("Saving eval metrics for testing set")
            utils.plot_precision_recall_curve(np_prec, np_rec, auc, model_name)
            utils.save_precision_recall_values(np_prec, np_rec, model_name)
            utils.save_eval_metrics(prec_at_100, prec_at_200, prec_at_300, prec_at_500, prec_at_1000, prec_at_2000,
                                    prec_at_all, auc, model_name)
            utils.save_labels_distribution_at_top_300_predictions(rel_dist_at_300, model_name)
            utils.save_relational_facts(self.facts, model_name)
            utils.save_sorted_pred_results(sorted_pred_result, model_name)

        return {'prec': np_prec, 'rec': np_rec, 'auc': auc, 'p@all': prec_at_all, 'p@100': prec_at_100,
                'p@200': prec_at_200, 'p@300': prec_at_300, 'p@500': prec_at_500, 'p@1000': prec_at_1000,
                'p@2000': prec_at_2000, 'rel_dist_at_300': rel_dist_at_300, 'relfacts': self.facts,
                'sorted_pred_results': sorted_pred_result, 'rel_pos_dist_at_300': rel_pos_dist_at_300}


def BagRELoader(path, rel2id, tokenizer, batch_size, shuffle, entpair_as_bag=False, bag_size=0, num_workers=0,
                collate_fn=BagREDataset.collate_fn):
    if bag_size == 0:
        collate_fn = BagREDataset.collate_fn
    else:
        collate_fn = BagREDataset.collate_bag_size_fn
    dataset = BagREDataset(path, rel2id, tokenizer, entpair_as_bag=entpair_as_bag, bag_size=bag_size)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True,
                                  num_workers=num_workers, collate_fn=collate_fn)
    return data_loader
