
import os
import time
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from ogb.graphproppred import Evaluator
from optims.scheduler.scheduler import LR_Scheduler


criterion = nn.CrossEntropyLoss()

# Multi-class Cross Entropy
# class of model optimizing & learning (ModelOptLearning)
class ModelOptLearning_MCE:
    def __init__(self, model, optimizer, 
                train_loader, valid_loader, test_loader,
                args):
        # initizing ModelOptLearning class
        self.model = model
        self.optimizer = optimizer

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        
        self.evaluator = Evaluator('ogbg-ppa')        
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
            y_true.append(labels.view(-1,1).detach().cpu())
            y_pred.append(torch.argmax(outputs.detach(), dim = 1).view(-1,1).cpu())

            total += len(labels)    

            loss = criterion(outputs.to(torch.float32), labels.view(-1,))

            total_loss += loss * len(labels)
                    
        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()

        input_dict = {'y_true': y_true, 'y_pred': y_pred}
        # eval results 
        rst = {}
        rst['loss'] = (1.0 * total_loss / total).item()
        rst['acc'] = self.evaluator.eval(input_dict)['acc']

        return rst
        
    def optimizing(self):
        scheduler = LR_Scheduler(self.optimizer, self.args.epochs, self.args.lr_warmup_type)
        # scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=80, gamma=0.6)

        valid_best = 0
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
                loss = criterion(outputs.to(torch.float32), labels.view(-1,))
                loss.backward()
                self.optimizer.step()

            train_rst = self.eval(self.model, self.train_loader)
            valid_rst = self.eval(self.model, self.valid_loader)
            test_rst = self.eval(self.model, self.test_loader)  

            train_loss = train_rst['loss']
            train_perf = train_rst['acc']
            valid_perf = valid_rst['acc']
            test_perf = test_rst['acc']

            eopch_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            eopch_time = time.time() - t0
            log_lr = {'lr': eopch_lr}
            log_time = {'time': eopch_time}
            print(f"epoch: {epoch}, train_loss {train_loss:.6f}, train perf {train_perf:.6f}, valid perf {valid_perf:.6f}, test perf {test_perf:.6f}, {eopch_lr}, {eopch_time:.2f}")
            logs_table = self.log_epoch(logs_table, train_rst, valid_rst, test_rst, log_lr, log_time)

            is_best_valid = bool((self.args.state_dict) & (valid_best < valid_perf) & (self.args.epoch_slice < epoch))
            valid_best = valid_perf
            if is_best_valid:
                if not os.path.exists(self.args.perf_dict_dir):
                    os.mkdir(self.args.perf_dict_dir)
                dict_file_path = os.path.join(self.args.perf_dict_dir, self.args.identity+'.pth')
                torch.save(self.model.state_dict(), dict_file_path)

            scheduler.step()
        
        if not os.path.exists(self.args.perf_xlsx_dir):
            os.mkdir(self.args.perf_xlsx_dir)
        logs_table.to_excel(os.path.join(self.args.perf_xlsx_dir, self.args.identity+'.xlsx'))
        if not os.path.exists(self.args.stas_xlsx_dir):
            os.mkdir(self.args.stas_xlsx_dir)
        

