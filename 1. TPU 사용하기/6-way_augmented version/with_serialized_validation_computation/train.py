from modules.utils import load_yaml, save_yaml, get_logger

from modules.earlystoppers import EarlyStopper
from modules.recorders import Recorder
from modules.datasets import CowDataset
from modules.datasets import HFlipedDataset,VFlipedDataset,RotatedDataset,BlurredDataset
from modules.trainer import Trainer

#from modules.preprocessor import get_preprocessor
from modules.optimizers import get_optimizer
from modules.metrics import get_metric
from modules.losses import get_loss

from models.utils import get_model

from torch.utils.data import DataLoader, ConcatDataset
import torch

from datetime import datetime, timezone, timedelta
import numpy as np
import random
import os
import copy

#for using tpu on pytorch
import torch_xla
import torch_xla.core.xla_model as xm 
import torch_xla.distributed.xla_multiprocessing as xmp # for multiprocessing

# Root Directory
PROJECT_DIR = os.path.dirname(__file__)

# Load config
config_path = os.path.join(PROJECT_DIR, 'config', 'train_config.yaml')
config = load_yaml(config_path)

# Train Serial
kst = timezone(timedelta(hours=9))
train_serial = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")

# Recorder Directory
RECORDER_DIR = os.path.join(PROJECT_DIR, 'results', 'train', train_serial)
os.makedirs(RECORDER_DIR, exist_ok=True)

# Data Directory
DATA_DIR = config['DIRECTORY']['dataset']

# Seed
torch.manual_seed(config['TRAINER']['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(config['TRAINER']['seed'])
random.seed(config['TRAINER']['seed'])

# GPU
os.environ['CUDA_VISIBLE_DEVICES'] = str(config['TRAINER']['gpu'])

# for serialized running
# at global scope...
SERIAL_EXEC = xmp.MpSerialExecutor()


if __name__ == '__main__':

    '''
    Set Logger
    '''
    logger = get_logger(name='train', dir_=RECORDER_DIR, stream=False)
    logger.info(f"Set Logger {RECORDER_DIR}")


    '''
    Load Data
    '''
    # Dataset
    original_dataset = CowDataset(img_folder = os.path.join(DATA_DIR, 'train', 'images'),
                              dfpath = os.path.join(DATA_DIR, 'train', 'grade_labels.csv')
                              )
    hf_dataset = HFlipedDataset(img_folder = os.path.join(DATA_DIR, 'train', 'images'),
                              dfpath = os.path.join(DATA_DIR, 'train', 'grade_labels.csv')
                              )
    vf_dataset = VFlipedDataset(img_folder = os.path.join(DATA_DIR, 'train', 'images'),
                              dfpath = os.path.join(DATA_DIR, 'train', 'grade_labels.csv')
                              )
    rt_dataset = RotatedDataset(img_folder = os.path.join(DATA_DIR, 'train', 'images'),
                              dfpath = os.path.join(DATA_DIR, 'train', 'grade_labels.csv')
                              )
    bl_dataset = BlurredDataset(img_folder = os.path.join(DATA_DIR, 'train', 'images'),
                              dfpath = os.path.join(DATA_DIR, 'train', 'grade_labels.csv')
                              )
    cc_dataset = CCroppedDataset(img_folder = os.path.join(DATA_DIR, 'train', 'images'),
                          dfpath = os.path.join(DATA_DIR, 'train', 'grade_labels.csv')
                          )   
    rc_dataset = RCroppedDataset(img_folder = os.path.join(DATA_DIR, 'train', 'images'),
                              dfpath = os.path.join(DATA_DIR, 'train', 'grade_labels.csv')
                              ) 

    

    # concatenate augmented train dataset
    train_dataset = ConcatDataset([original_dataset,hf_dataset,vf_dataset,rt_dataset,bl_dataset, cc_dataset, rc_dataset])
    val_dataset = CowDataset(img_folder = os.path.join(DATA_DIR, 'val', 'images'),
                             dfpath = os.path.join(DATA_DIR, 'val', 'grade_labels.csv'))


    '''
    Set trainer
    '''
 
    # Loss
    loss = get_loss(loss_name=config['TRAINER']['loss'])
    
    # Metric
    metrics = {metric_name: get_metric(metric_name) for metric_name in config['TRAINER']['metric']}
    
    # Early stoppper
    early_stopper = EarlyStopper(patience=config['TRAINER']['early_stopping_patience'],
                                mode=config['TRAINER']['early_stopping_mode'],
                                logger=logger)

    # Load model
    model_name = config['TRAINER']['model']
    model_args = config['MODEL'][model_name]

    # wrapping model for optimized memory usage in 'fork' method
    wrapped_model = xmp.MpModelWrapper(get_model(model_name = model_name, model_args = model_args))

    def serial_early_stopping(flags:dict):

        """
        Validation
        """
 
        ######################################
        val_dataloader = flags['val_dataloader']
        row_dict = flags['row_dict']
        trainer = flags['trainer']
        epoch_index = flags['epoch_index']
        n_epochs = flags['n_epochs']
        index = flags['device_index']
        recorder = flags['recorder'] 
        ######################################
        
        #print(f"Val {epoch_index}/{n_epochs}")
        xm.master_print(f"--Val {epoch_index}/{n_epochs} -- device_num:{index}")      
        logger.info(f"--Val {epoch_index}/{n_epochs} -- device_num:{index}")


        
        trainer.train(dataloader=val_dataloader, epoch_index=epoch_index, mode='val')
        
        row_dict['val_loss'] = trainer.loss_mean
        row_dict['val_elapsed_time'] = trainer.elapsed_time 
        
        for metric_str, score in trainer.score_dict.items():
            row_dict[f"val_{metric_str}"] = score
            

        recorder.add_row(row_dict)
        recorder.save_plot(config['LOGGER']['plot'])
        

        """
        Early stopper
        """
        early_stopping_target = config['TRAINER']['early_stopping_target']
        early_stopper.check_early_stopping(loss=row_dict[early_stopping_target])

        if (early_stopper.patience_counter == 0) or (epoch_index == n_epochs-1):
            recorder.save_weight(epoch=epoch_index)
            best_row_dict = copy.deepcopy(row_dict)
            
        if early_stopper.stop == True:
            logger.info(f"Eearly stopped, counter {early_stopper.patience_counter}/{config['TRAINER']['early_stopping_patience']}")

        return

    
    def map_fn(index, args):

        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #calling xla_device should be in local scope, so below code line belong to local scope
        device = xm.xla_device() #instead use xm.xla_device()



        #define data(train, val) sampler for parallel processing with TPU
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True)
        
        # define DataLoader
        # because sampler option is mutually exclusive with shuffle, shuffle should be False in train_config file
        train_dataloader = DataLoader(dataset = train_dataset,
                                      batch_size = config['DATALOADER']['batch_size'],
                                      num_workers = config['DATALOADER']['num_workers'],
                                      shuffle = config['DATALOADER']['shuffle'],
                                      pin_memory = config['DATALOADER']['pin_memory'],
                                      drop_last = config['DATALOADER']['drop_last'],
                                      sampler = train_sampler
                                      )

        
        val_dataloader = DataLoader(dataset = val_dataset,
                                    batch_size = config['DATALOADER']['batch_size'],
                                    num_workers = config['DATALOADER']['num_workers'], 
                                    shuffle = False,
                                    pin_memory = config['DATALOADER']['pin_memory'],
                                    drop_last = config['DATALOADER']['drop_last'],
                                    )
        
        logger.info(f"Load data, train:{len(train_dataset)} val:{len(val_dataset)}--device_num:{index}")


        '''
        Set model
        '''
        # allocate computation to model..
        model = wrapped_model.to(device)

        # Optimizer
        optimizer = get_optimizer(optimizer_name=config['TRAINER']['optimizer'])
        optimizer = optimizer(params=model.parameters(),lr=config['TRAINER']['learning_rate'])

        

        # AMP
        if config['TRAINER']['amp'] == True:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        
        # Trainer
        trainer = Trainer(model=model,
                          optimizer=optimizer,
                          loss=loss,
                          metrics=metrics,
                          device=device,
                          logger=logger,
                          amp=amp if config['TRAINER']['amp'] else None,
                          device_index = index,
                          interval=config['LOGGER']['logging_interval'])
        '''
        Logger
        '''    
        # Recorder
        recorder = Recorder(record_dir=RECORDER_DIR,
                            model=model,
                            optimizer=optimizer,
                            scheduler=None,
                            amp=amp if config['TRAINER']['amp'] else None,
                            logger=logger)

        # Save train config
        save_yaml(os.path.join(RECORDER_DIR, 'train_config.yml'), config)

        '''
        TRAIN
        '''
        # Train
        n_epochs = config['TRAINER']['n_epochs']
        epoch_ind = 0
        for epoch_index in range(n_epochs):

            # Set Recorder row
            row_dict = dict()
            row_dict['epoch_index'] = epoch_index
            row_dict['train_serial'] = train_serial
                
            """
            Train
            """
            # instead of use python built-in function, we use master_print()
            # print(f"Train {epoch_index}/{n_epochs}")
            xm.master_print(f"Train {epoch_index}/{n_epochs} -- device_num:{index}")       
            logger.info(f"--Train {epoch_index}/{n_epochs} -- device_num:{index}")
            
            trainer.train(dataloader=train_dataloader, epoch_index=epoch_index, mode='train')
            
            row_dict['train_loss'] = trainer.loss_mean
            row_dict['train_elapsed_time'] = trainer.elapsed_time 
            
            for metric_str, score in trainer.score_dict.items():
                row_dict[f"train_{metric_str}"] = score
                
            trainer.clear_history()

            dic = {'epoch_index':epoch_index, 'trainer':trainer, 'n_epochs':n_epochs,
                   'device_index' : index, 'recorder':recorder, 'row_dict':row_dict,
                   'val_dataloader':val_dataloader}
            
            SERIAL_EXEC.run(lambda: serial_early_stopping(dic))
            
            trainer.clear_history()
            logger.info('---- next train step ---- ')
            

    #execution
    flags = {}
    xmp.spawn(map_fn, args = (flags, ), nprocs=8, start_method='fork') 
