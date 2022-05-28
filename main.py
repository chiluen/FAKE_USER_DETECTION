import os, pickle, json, datetime, yaml
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from transformers import AutoTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from argparse import ArgumentParser
from random import sample

from util.data_loader_generator import data_loader_process
from util.metrics import accuracy
import wandb
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--mode', type = str, default='train', help="Select from train or test")
parser.add_argument('--model', type=str, default='bert', help="Choose['bert']")
parser.add_argument('--config', type = str, default='./config.yaml', help="Select config")
parser.add_argument('--cpu', action='store_true', help="Use cpu to train")
parser.add_argument('--ckpt', type = str, default='', help="Enter the path of checkpoint")

args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
    
#=======================#
#=====Training mode=====#
#=======================#
if args.mode == 'train':
    
    now = datetime.datetime.now()
    now = now.strftime("%d-%m-%Y-%H:%M:%S")

    ckpt_address = './ckpt/' + now
    log_address = './log/' + now

    os.mkdir(ckpt_address)
    os.mkdir(log_address)
    
    #=====Construct model=====#
    print("Construct model")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 2)
    
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    LEARNING_RATE = config["TRAIN"]["LEARNING_RATE"]
    WEIGHT_DECAY = config["TRAIN"]["WEIGHT_DECAY"]
    EPSILON = config["TRAIN"]["EPSILON"]
    EPOCHS = config["TRAIN"]["EPOCHS"]

    optimizer = AdamW(model.parameters(), lr = LEARNING_RATE, eps = EPSILON)
    
    #Get log record
    wandb.login()
    wandb.init(project="CNS_Final", config=config)

    #===Construct dataloader===#
    train_dataloader = data_loader_process(path = config["TRAIN"]["TARGET_PATH"], second_path = config["TRAIN"]["SUPPORT_PATH"], mode="train")
    test_dataloader = data_loader_process(path = config["TEST"]["TARGET_PATH"], second_path = config["TEST"]["SUPPORT_PATH"], mode="test")

    #=====Training script=====#
    print("Start training")
    model.train()
    for epoch in range(EPOCHS):

        model.train()
        record_loss = []
        record_acc = []

        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            optimizer.zero_grad()
            output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss, logits = output[0], output[1]
            acc = accuracy(logits, b_labels)

            record_loss.append(loss.item())
            record_acc.append(acc)
            loss.backward()
            optimizer.step()

            #record every 100 steps
            if step % 100 == 0 and step != 0:
                loss = torch.FloatTensor(record_loss).mean().item()
                acc = torch.FloatTensor(record_acc).mean().item()
                wandb.log({"training_loss": loss})
                wandb.log({"training_acc": acc})
                record_loss = []
                record_acc = []

                print("Epoch: {} | steps: {} | train_acc: {} | train_loss: {}".format(epoch, step, acc, loss), flush=True)
        
        # #Start to do evaluation after one EPOCH
        model.eval()
        record_loss = []
        record_acc = []
        record_output = []
        import ipdb; ipdb.set_trace()
        for batch in test_dataloader:

            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss, logits = output[0], output[1]
                record_acc.append(accuracy(logits, b_labels))
                record_loss.append(loss.item())

                #record logs
                record_output.extend(torch.argmax(logits, axis=1).tolist())

        acc = torch.FloatTensor(record_acc).mean().item()
        loss = torch.FloatTensor(record_loss).mean().item()
        wandb.log({"evaluation_acc": acc})
        wandb.log({"evaluation_loss": loss})
        print("Testing! Epoch: {} | test_acc: {} | test_loss: {}".format(epoch, acc, loss))

        #save ckpt
        new_ckpt = ckpt_address + '/Epoch_{}_acc_{}'.format(epoch, round(acc, 2))
        os.mkdir(new_ckpt)
        model.save_pretrained(new_ckpt)

        #save losg
        new_log = log_address + '/Epoch_{}_acc_{}.pickle'.format(epoch, round(acc, 5))
        with open(new_log, 'wb') as f:
            pickle.dump(record_output, f)