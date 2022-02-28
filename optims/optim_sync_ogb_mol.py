

import os
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
from cmath import inf
from sklearn import metrics 
from ogb.graphproppred import Evaluator
from torch.optim.lr_scheduler import LambdaLR

cls_criterion = nn.BCEWithLogitsLoss()
reg_criterion = nn.MSELoss()

class LinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, t_total, warmup_steps=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(LinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


# class of model optimizing & learning (ModelOptLearning)
class ModelOptLearning_OGB_HIV:
    def __init__(self, model, optimizer, 
                train_loader, valid_loader, test_loader,
                args):
        # initizing ModelOptLearning class
        self.model = model
        self.optimizer = optimizer

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.evaluator = Evaluator(args.dataset)        
        self.args = args

    def log_epoch(self, logs_table, train_rst, valid_rst, test_rst, log_lr, log_time):
        table_head = []
        table_data = []
        for keys in train_rst.keys():
            table_head.append(f'train-{keys}')
            table_data.append(train_rst[keys])
        for keys in valid_rst.keys():
            table_head.append(f'valid-{keys}')
            table_data.append(valid_rst[keys])
        for keys in test_rst.keys():
            table_head.append(f'test-{keys}')
            table_data.append(test_rst[keys])
        for keys in log_lr.keys():
            table_head.append(f'{keys}')
            table_data.append(log_lr[keys])
        for keys in log_time.keys():
            table_head.append(f'{keys}')
            table_data.append(log_time[keys])
        
        return logs_table.append(pd.DataFrame([table_data], columns=table_head), ignore_index=True)

    def eval(self, model, loader):
        model.eval()
        total, total_loss = 0, 0  
        y_true, y_pred = [], []
        
        for graphs, labels in loader:    
            graphs, labels = graphs.to(self.args.device), labels.to(self.args.device)
            nfeats = graphs.ndata['feat']
            efeats = graphs.edata['feat']
            with torch.no_grad():
                outputs = model(graphs, nfeats, efeats)
            y_true.append(labels.view(outputs.shape).detach().cpu())
            y_pred.append(outputs.detach().cpu())

            total += len(labels)    
            is_labeled = labels == labels
            if "classification" in self.args.task_type: 
                loss = cls_criterion(outputs.to(torch.float32)[is_labeled], labels.to(torch.float32)[is_labeled])
            else :
                loss = reg_criterion(outputs.to(torch.float32)[is_labeled], labels.to(torch.float32)[is_labeled])
            total_loss += loss * len(labels)
                    
        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()

        input_dict = {'y_true': y_true, 'y_pred': y_pred}
        # eval results 
        rst = {}
        rst['loss'] = (1.0 * total_loss / total).item()
        rst[self.args.eval_metric] = self.evaluator.eval(input_dict)[self.args.eval_metric]

        return rst 
        
    def optimizing(self):
        scheduler = LinearSchedule(self.optimizer, self.args.epochs)
        # scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        valid_best_cls = 0
        valid_best_reg = inf
        logs_table = pd.DataFrame()
        for epoch in range(self.args.epochs):
            # training model 
            self.model.train()
            t0 = time.time()
            for _, batch in enumerate(tqdm(self.train_loader, desc='Iteration')):              
            # for graphs, labels in self.train_loader:
                graphs, labels = batch
                graphs, labels = graphs.to(self.args.device), labels.to(self.args.device)
                nfeats = graphs.ndata['feat']
                efeats = graphs.edata['feat']

                outputs = self.model(graphs, nfeats, efeats)
                self.optimizer.zero_grad()
                is_labeled = labels == labels
                if "classification" in self.args.task_type: 
                    loss = cls_criterion(outputs.to(torch.float32)[is_labeled], labels.to(torch.float32)[is_labeled])
                else:
                    loss = reg_criterion(outputs.to(torch.float32)[is_labeled], labels.to(torch.float32)[is_labeled])
                loss.backward()
                self.optimizer.step()

            train_rst = self.eval(self.model, self.train_loader)
            valid_rst = self.eval(self.model, self.valid_loader)
            test_rst = self.eval(self.model, self.test_loader)  

            train_loss = train_rst['loss']
            train_perf = train_rst[self.args.eval_metric]
            valid_perf = valid_rst[self.args.eval_metric]
            test_perf = test_rst[self.args.eval_metric]

            eopch_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            eopch_time = time.time() - t0
            log_lr = {'lr': eopch_lr}
            log_time = {'time': eopch_time}
            print(f"epoch: {epoch}, train_loss {train_loss:.6f}, train perf {train_perf:.6f}, valid perf {valid_perf:.6f}, test perf {test_perf:.6f}, {eopch_lr}, {eopch_time:.2f}")
            logs_table = self.log_epoch(logs_table, train_rst, valid_rst, test_rst, log_lr, log_time)
            scheduler.step()

            if "classification" in self.args.task_type:
                is_best_valid = bool((self.args.state_dict) & (valid_best_cls < valid_perf) & (self.args.epoch_slice < epoch))
                valid_best_cls = valid_perf
            else: 
                is_best_valid = bool((self.args.state_dict) & (valid_best_reg > valid_perf) & (self.args.epoch_slice < epoch))
                valid_best_reg = valid_perf
            if is_best_valid:
                if not os.path.exists(self.args.dict_dir):
                    os.mkdir(self.args.dict_dir)
                dict_file_path = os.path.join(self.args.dict_dir, self.args.identity+'.pth')
                torch.save(self.model.state_dict(), dict_file_path)
        
        if not os.path.exists(self.args.xlsx_dir):
            os.mkdir(self.args.xlsx_dir)
        logs_table.to_excel(os.path.join(self.args.xlsx_dir, self.args.identity+'.xlsx'))
        

