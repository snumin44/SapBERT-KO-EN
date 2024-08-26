import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, XLMRobertaTokenizerFast, XLMRobertaConfig

class Dataset_CSV(Dataset):

    def __init__(self, sent0, sent1, label):
        self.sent0 = sent0
        self.sent1 = sent1
        self.label = label

    @classmethod
    def load_dataset(cls, data_path, delimiter='\t'):
        df = pd.read_csv(data_path, sep=delimiter)
        
        sent0 = df['sent0'].to_list()
        sent1 = df['sent1'].to_list()
        label = df['label'].to_list()

        # map cui into unique integer
        unique_label = sorted(set(label))
        label_to_int = {label: idx for idx, label in enumerate(unique_label)}
        mapped_label = [label_to_int[lab] for lab in label]

        return cls(sent0, sent1, mapped_label)


    def __len__(self):
        assert len(self.sent0) == len(self.sent1)
        assert len(self.sent0) == len(self.label)
        return len(self.sent0)

    def __getitem__(self, index):
        return {'sent0':self.sent0[index],
                'sent1':self.sent1[index],
                'label':self.label[index]}}


class DataCollator(object):
    
    def __init__(self, args):        
        if 'xlm-rberta' in args.model:
            self.config = XLMRobertaConfig.from_pretrained(args.model)
            self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(args.tokenizer) # config=self.config 제거
        else:
            self.config = AutoConfig.from_pretrained(args.model)
            self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer,
                                                           config=self.config)
        
        self.padding = args.padding
        self.max_length = args.max_length
        self.truncation = args.truncation

    def __call__(self, samples):
        sent0_lst = []
        sent1_lst = []
        label_lst = []
        for sample in samples:
            sent0_lst.append(sample['sent0'])        
            sent1_lst.append(sample['sent1'])
            label_lst.append(sample['label'])
            
        sents_num = len(samples)
        sents_lst = sent0_lst + sent1_lst

        # Encode all sentences at once as a single encoder. 
        sent_features = self.tokenizer(
            sents_lst,
            padding = self.padding,
            max_length = self.max_length,
            truncation = self.truncation
            )
        
        features = {}
        for key in sent_features:
            features[key] = [[sent_features[key][i],
                              sent_features[key][i+sents_num]] for i in range(sents_num)]            
                
        batch = {
            'input_ids':torch.tensor(features['input_ids']),
            'attention_mask':torch.tensor(features['attention_mask']),
            'label': torch.tensor(label_lst)  
            } 
        
        return batch