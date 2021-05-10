import torch
import numpy
from torch import nn, optim
from redsandt.data_loader.data_loader import BagRELoader
from .utils import AverageMeter
from tqdm import tqdm


class BagRE(nn.Module):

    def __init__(self, model, train_path, val_path, test_path, model_name, ckpt, batch_size=32, max_epoch=100, lr=0.1,
                 weight_decay=1e-5, warmup_step_ratio=0.1, opt='sgd', bag_size=0, weighted_loss=False):
        super().__init__()

        self.model_name = model_name
        self.max_epoch = max_epoch
        self.bag_size = bag_size
        self.warmup_step_ratio = warmup_step_ratio
        weighted_loss = True if weighted_loss == "True" else False

        # Load data
        if train_path != None:
            self.train_loader = BagRELoader(train_path, model.rel2id, model.sentence_encoder.tokenize, batch_size, True,
                                            bag_size=bag_size, entpair_as_bag=False)

        if val_path != None:
            self.val_loader = BagRELoader(val_path, model.rel2id, model.sentence_encoder.tokenize, batch_size, False,
                                          bag_size=bag_size, entpair_as_bag=True)

        if test_path != None:
            self.test_loader = BagRELoader(test_path, model.rel2id, model.sentence_encoder.tokenize, batch_size, False,
                                           bag_size=bag_size, entpair_as_bag=True)

        # Model
        self.model = nn.DataParallel(model)

        # Criterion
        if weighted_loss:
            self.criterion = nn.CrossEntropyLoss(weight=self.train_loader.dataset.weight, reduction='mean')
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Params and optimizer
        params = self.model.parameters()
        self.lr = lr
        if opt == 'sgd':
            self.optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
        elif opt == 'adamw':
            from transformers import AdamW
            params = list(self.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            grouped_params = [
                {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay,
                 'lr': lr, 'ori_lr': lr},
                {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': lr,
                 'ori_lr': lr}]
            self.optimizer = AdamW(grouped_params, correct_bias=False)
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'adamw'.")

        if self.warmup_step_ratio > 0:
            from transformers import get_cosine_schedule_with_warmup
            training_steps = self.train_loader.dataset.__len__() // batch_size * self.max_epoch
            warmup_steps = numpy.ceil(self.warmup_step_ratio * training_steps)
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps,
                                                             num_training_steps=training_steps)
        else:
            self.scheduler = None
        if torch.cuda.is_available():
            self.cuda()

        self.ckpt = ckpt

    def train_model(self):
        best_auc = 0

        # Loop over epochs
        for epoch in range(self.max_epoch):
            # -------------------------------
            # ---------- Training -----------
            # -------------------------------
            # Prepare model for training
            self.train()
            print("\n\n=== Epoch %d train ===" % epoch)
            avg_train_loss = AverageMeter()
            avg_train_acc = AverageMeter()
            avg_pos_train_acc = AverageMeter()

            # Iterate through the dataset
            t = tqdm(self.train_loader)
            for iter, data in enumerate(t):
                # Convert inputs to cuda tensors
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]  # (B)
                bag_name = data[1]  # (B)
                scope = data[2]  # (B)
                args = data[3:]  # input_ids, att_mask

                # Obtain the logits from the model (B, N)
                logits = self.model(label, scope, *args, bag_size=self.bag_size)

                # Compute the loss
                loss = self.criterion(logits, label)

                ## Record training scores (loss, acc, pos_acc)
                # Get (score, pred) to compute log scores
                score, pred = logits.max(-1)  # (B)
                acc = float((pred == label).long().sum()) / label.size(0)
                pos_total = (label != 0).long().sum()
                pos_correct = ((pred == label).long() * (label != 0).long()).sum()
                if pos_total > 0:
                    pos_acc = float(pos_correct) / float(pos_total)
                else:
                    pos_acc = 0

                # Update loggings
                avg_train_loss.update(loss.item(), 1)
                avg_train_acc.update(acc, 1)
                avg_pos_train_acc.update(pos_acc, 1)
                t.set_postfix(loss=avg_train_loss.avg, acc=avg_train_acc.avg, pos_acc=avg_pos_train_acc.avg)

                # Back-propagate the gradients
                loss.backward()

                # Update parameters and take a step using the computed gradient
                self.optimizer.step()

                # Clear gradients
                self.optimizer.zero_grad()

                # Update the learning rate
                if self.warmup_step_ratio > 0:
                    self.scheduler.step()

            # -------------------------------
            # --------- Validation ----------
            # -------------------------------
            print("\n\n=== Epoch %d val ===" % epoch)
            result = self.eval_model(self.val_loader)
            print("auc: %.4f" % result['auc'])
            print("Previous best auc on val set: %f" % (best_auc))
            print("P@100:", result['p@100'])
            print("P@200:", result['p@200'])
            print("P@300:", result['p@300'])
            print("P@2000:", result['p@2000'])
            print('\nRelation Distribution on Top 300 predictions:')
            for key, value in result['rel_dist_at_300'].items():
                print(key, ": ", value)

            if result['auc'] > best_auc:
                print("Best ckpt and saved.")
                torch.save({'state_dict': self.model.module.state_dict()}, self.ckpt)
                best_auc = result['auc']

    def eval_model(self, eval_loader, save_eval_metrics=False):
        # Prepare model for evaluation
        self.model.eval()
        with torch.no_grad():
            t = tqdm(eval_loader)
            pred_result = []
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                bag_name = data[1]
                scope = data[2]
                args = data[3:]

                # Compute predicted outputs
                logits = self.model(None, scope, *args, train=False, bag_size=self.bag_size)

                for i in range(logits.size(0)):
                    for relid in range(self.model.module.num_class):
                        if self.model.module.id2rel[relid] != 'NA':
                            pred_result.append({'entpair': bag_name[i][:2], 'relation': self.model.module.id2rel[relid],
                                                'score': logits[i][relid].item()})

            result = eval_loader.dataset.eval(pred_result, self.model_name, save_eval_metrics)

        return result

    def load_state_dict(self, state_dict):
        self.model.module.load_state_dict(state_dict, strict=False)
        self.model.module.to(torch.device("cuda"))
