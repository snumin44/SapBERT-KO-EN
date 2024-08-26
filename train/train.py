import os
import re
import time
import random

import datetime
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader 
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast 
from torch.cuda.amp import GradScaler
from datasets import load_dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

import sys
sys.path.append("../") 

from src.data_loader import Dataset_CSV, DataCollator
from src.model import SimCSE

LOGGER = logging.getLogger()

def argument_parser():

    parser = argparse.ArgumentParser(description='train simcse')

    # Required
    parser.add_argument('--model', type=str, required=True,
                        help='Directory of pretrained model'
                       )    
    parser.add_argument('--tokenizer', type=str, required=True,
                        help='Directory of pretrained tokenizer'
                       )     
    parser.add_argument('--train_data', type=str, required=True,
                        help='Directory of Training dataset'
                       )
    parser.add_argument('--output_path', type=str, default='../output/checkpoint',
                        help='Directory for output'
                       )
    
    # Tokenizer & Collator settings
    parser.add_argument('--max_length', default=64, type=int,
                        help='Max length of sequence'
                       )
    parser.add_argument('--padding', action="store_false", default=True,
                        help='Add padding to short sentences'
                       )
    parser.add_argument('--truncation', action="store_false", default=True,
                        help='Truncate extra tokens when exceeding the max_length'
                       )
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size'
                       )
    parser.add_argument('--shuffle', action="store_false", default=True,
                        help='Load shuffled sequences'
                       )

    # Multi-Similarity Loss
    parser.add_argument('--threshold', default=0.3, type=float,
                        help='Threshold for Multi-Similarity Loss'
                       )
    parser.add_argument('--scale_pos', default=60, type=float,
                        help='Scale for positive sample'
                       )
    parser.add_argument('--scale_neg', default=1, type=float,
                        help='Scale for negative sample'
                       )
    parser.add_argument('--margin', default=0.3, type=float,
                        help='Margin for Multi-Similarity Loss'
                       )
    parser.add_argument('--hard_pair_mining', action='store_true', default=False,
                        help='Calculate loss with hard pairs only'
                       )
    
    # Train config    
    parser.add_argument('--epochs', default=1, type=int,
                        help='Training epochs'
                       )       
    parser.add_argument('--pooler', default='cls', type=str,
                        help='Pooler type : {pooler_output|cls|mean|max}'
                       )    
    parser.add_argument('--weight_decay', default=1e-2, type=float,
                        help='Weight decay'
                       )       
    parser.add_argument('--no_decay', nargs='+', default=['bias', 'LayerNorm.weight'],
                        help='List of parameters to exclude from weight decay' 
                       )         
    parser.add_argument('--temp', default=0.5, type=float,
                        help='Temperature for similarity'
                       )       
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Drop-out ratio'
                       )       
    parser.add_argument('--learning_rate', default=5e-5, type=float,
                        help='Leraning rate'
                       )       
    parser.add_argument('--eta_min', default=0, type=int,
                        help='Eta min for CosineAnnealingLR scheduler'
                       )   
    parser.add_argument('--eps', default=1e-8, type=float,
                        help='Epsilon for AdamW optimizer'
                       )   
    parser.add_argument('--amp', action="store_true", default=False,
                        help='Use Automatic Mixed Precision for training'
                       ) 
    parser.add_argument('--eval_step', default=500, type=int,
                        help='Evaluaton interval when eval_strategy is set to <steps>'
                       )   
    parser.add_argument('--device', default = 'cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help = 'Choose a type of device for training'
                       )
    parser.add_argument('--random_seed', default = 42, type=int,
                        help = 'Random seed'
                       )  
    args = parser.parse_args()
    return args

def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]','%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

def seed_everything(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    os.environ["PYTHONHASHSEED"] = str(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_adamw_optimizer(model, args):
    if args.no_decay: 
        # skip weight decay for some specific parameters i.e. 'bias', 'LayerNorm'.
        no_decay = args.no_decay  
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    else:
        # weight decay for every parameter.
        optimizer_grouped_parameters = model.parameters()
    optimizer = AdamW(optimizer_grouped_parameters, lr = args.learning_rate, eps = args.eps)
    return optimizer

def get_scheduler(optimizer, args):
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min, last_epoch=-1)
    return scheduler

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed))) # Round to the nearest second.
    return str(datetime.timedelta(seconds=elapsed_rounded)) # Format as hh:mm:ss 

def train(encoder, train_dataloader, optimizer, scheduler, scaler, args):
    best_score = None
    total_train_loss = 0

    t0 = time.time()

    for epoch_i in range(args.epochs):           

        LOGGER.info(f'Epoch : {epoch_i+1}/{args.epochs}')
        
        encoder.train()
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            
            # pass the data to device(cpu or gpu)            
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            label = batch['label'].to(args.device)

            optimizer.zero_grad()

            if args.amp:
                train_loss = encoder(input_ids, attention_mask, label)
                scaler.scale(train_loss.mean()).backward()

                # Clip the norm of the gradients to 5.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0) 
                
                scaler.step(optimizer)
                scaler.update()            

            else:
                train_loss = encoder(input_ids, attention_mask, label)

                # Clip the norm of the gradients to 5.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0)  

                optimizer.step()
            
            scheduler.step()
            
            if isinstance(encoder, nn.DataParallel):
                model_to_save = encoder.module
            else:
                model_to_save = encoder
            
            if (step+1) % args.eval_step == 0:
                epoch_c = round((step + 1) / len(train_dataloader), 2) 
                print(f'Epoch:{epoch_c}, Train Loss:{train_loss / len(train_dataloader):4f}')

    avg_train_loss = total_train_loss / (len(train_dataloader) * args.epochs)
    training_time = format_time(time.time() - t0)
    
    print(f"Training Time: {training_time}, Average Training Loss: {avg_train_loss}")
    LOGGER.info(f'>>> Save the checkpoint in {args.output_path}.')                    
    model_to_save.save_model(args.output_path)


def main(args):
    init_logging()
    seed_everything(args)
    
    LOGGER.info('*** Train SAP Bert ***')
    train_dataset = Dataset_CSV.load_dataset(args.train_data)
       
    collator = DataCollator(args)
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=args.shuffle,
                                  collate_fn=collator)

    if args.device == 'cuda':
        LOGGER.info("Using nn.DataParallel")
        encoder = SimCSE(args).to(args.device)
        encoder = torch.nn.DataParallel(encoder)
        
    else:
        encoder = SimCSE(args).to(args.device)

    optimizer = get_adamw_optimizer(encoder, args)
    scheduler = get_scheduler(optimizer, args)

    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    # Train!
    torch.cuda.empty_cache()
    train(encoder, train_dataloader, optimizer, scheduler, scaler, args)

if __name__ == '__main__':
    args = argument_parser()
    main(args)