from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
from random import sample
import pickle

def data_loader_process(path=None, second_path = None, mode="train"): 
    """
    path: target path
    second_path: support path
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  

    if mode == "train":
        print("Construct for train dataset")
        with open(path ,'rb') as f:
            positive_data = pickle.load(f)

        with open(second_path ,'rb') as f:
            negative_data = pickle.load(f)

        # testing 原本是//4
        negative_data = negative_data[:len(negative_data)//4]
        positive_data = positive_data[:len(negative_data)]

        negative_data = [x for x in negative_data if x == x] #delete nan
        positive_data = [x for x in positive_data if x == x] #delete nan

        positive_data = tokenizer(positive_data, padding= 'max_length', max_length=30, truncation=True)
        negative_data = tokenizer(negative_data, padding= 'max_length', max_length=30, truncation=True)

        #deal with dataloader construction
        positive_len = len(positive_data['input_ids'])
        negative_len = len(negative_data['input_ids'])

        train_label = [1 for i in range(positive_len)]
        train_label.extend([0 for i in range(negative_len)])

        train_input_ids = positive_data['input_ids']
        train_input_ids.extend(negative_data['input_ids'])

        train_attention_mask = positive_data['attention_mask']
        train_attention_mask.extend(negative_data['attention_mask'])

        batch_size = 20
        train_data = TensorDataset(torch.tensor(train_input_ids), torch.tensor(train_attention_mask), torch.tensor(train_label))
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
        return train_dataloader
    else:
        print("Construct for test dataset") 
        with open(path ,'rb') as f:
            positive_data = pickle.load(f)

        with open(second_path ,'rb') as f:
            negative_data = pickle.load(f)


        negative_data = [x for x in negative_data if x == x] #delete nan
        positive_data = [x for x in positive_data if x == x] #delete nan

        positive_data = tokenizer(positive_data, padding= 'max_length', max_length=30, truncation=True)
        negative_data = tokenizer(negative_data, padding= 'max_length', max_length=30, truncation=True)

        #deal with dataloader construction
        positive_len = len(positive_data['input_ids'])
        negative_len = len(negative_data['input_ids'])

        test_label = [1 for i in range(positive_len)]
        test_label.extend([0 for i in range(negative_len)])

        test_input_ids = positive_data['input_ids']
        test_input_ids.extend(negative_data['input_ids'])

        test_attention_mask = positive_data['attention_mask']
        test_attention_mask.extend(negative_data['attention_mask'])

        batch_size = 20
        test_data = TensorDataset(torch.tensor(test_input_ids), torch.tensor(test_attention_mask), torch.tensor(test_label))
        test_dataloader = DataLoader(test_data, batch_size=batch_size)
        return test_dataloader