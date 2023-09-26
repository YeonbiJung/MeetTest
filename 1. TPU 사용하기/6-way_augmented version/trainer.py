import numpy as np
import torch

from time import time
from tqdm import tqdm


import torch_xla.core.xla_model as xm 
import torch_xla.distributed.parallel_loader as pl # for parallel loading


class Trainer():
    
    def __init__(self, model, optimizer, loss, metrics, device, logger, amp, device_index, interval=100):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.device = device # in train.py, device should be xm.xla_device()
        
        self.logger = logger
        self.amp = amp
        self.interval = interval
        
        # History
        self.loss_sum = 0 # Epoch loss sum
        self.loss_mean = 0 # epoch loss mean
        self.filenames = list()
        self.y = list()
        self.y_preds = list()
        self.score_dict = dict()
        self.elapsed_time = 0
        self.device_index = device_index
        
    def train(self, mode, dataloader, epoch_index=0):
        start_timestamp = time()
        self.model.train() if mode == 'train' else self.model.eval()

        para_loader = pl.ParallelLoader(dataloader, [self.device], loader_prefetch_size=8, device_prefetch_size=4).per_device_loader(self.device)
        #mp_device_loader = pl.MpDeviceLoader(dataloader, self.device)
        for batch_index, (x,y) in enumerate(para_loader):
        
        #for batch_index, (x,y) in mp_device_loader:
            x,y = x.to(self.device, dtype=torch.float), y.to(self.device, dtype=torch.long)
            
            # Inference
            y_pred = self.model(x)
            
            # Loss
            loss = self.loss(y_pred, y)
            
            # Update
            if mode == 'train':
                self.optimizer.zero_grad()
                
                if self.amp is None:
                    loss.backward()
                
                else:
                    with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                    
                #self.optimizer.step()
                xm.optimizer_step(self.optimizer)
                
            elif mode in ['val','test']:
                pass
            else:
                raise ValueError('Mode should be either train, val, or test')
            
            # History
            #self.filenames += filename
            self.loss_sum += loss.item()
            self.y_preds.append(y_pred.argmax(dim=1))
            self.y.append(y)
            
            # Logging
            if batch_index % self.interval == 0:
                msg = f"batch: {batch_index}/{len(dataloader)} loss: {loss.item()} - device_num : {self.device_index}"
                self.logger.info(msg)
                
        # Epoch history
        self.loss_mean = self.loss_sum / len(dataloader)
        
        # Metric
        self.y_preds = torch.cat(self.y_preds, dim=0).cpu().tolist()
        self.y = torch.cat(self.y, dim=0).cpu().tolist()
        
        for metric_name, metric_func in self.metrics.items():
            score = metric_func(self.y, self.y_preds)
            self.score_dict[metric_name] = score
        
        # Elapsed time
        end_timestamp = time()
        self.elapsed_time = end_timestamp - start_timestamp
        
    def clear_history(self):
        self.loss_sum = 0
        self.loss_mean = 0
        self.y_preds = list()
        self.y = list()
        self.score_dict = dict()
        self.elapsed_time = 0
                
                
