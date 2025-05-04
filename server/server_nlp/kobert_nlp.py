import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp
import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook
#kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
#transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split

device = torch.device("cuda:0")
bertmodel, vocab = get_pytorch_kobert_model()
data_path = 'docent_dataset.xlsx'

# Setting parameters
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

#토큰화
tokenizer = get_tokenizer()
tok = gluonnlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = gluonnlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=2,   ##클래스 수 조정##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

def set_dataset(data_path):
    docent_data = pd.read_excel(data_path)

    docent_data.loc[(docent_data['Emotion'] == "길안내"), 'Emotion'] = 0
    docent_data.loc[(docent_data['Emotion'] == "질문"), 'Emotion'] = 1

    data_list = []
    for q, label in zip(docent_data['Sentence'], docent_data['Emotion'])  :
        data = []
        data.append(q)
        data.append(str(label))

        data_list.append(data)

    #train & test 데이터로 나누기                                    
    dataset_train, dataset_test = train_test_split(data_list, test_size=0.25, random_state=0)
    
    #토큰화
    tokenizer = get_tokenizer()
    tok = gluonnlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
    data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)

    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)
    return train_dataloader, test_dataloader

def make_train_model():
    # 데이터셋
    dataloaders = set_dataset(data_path=data_path)
    train_dataloader = dataloaders[0]
    test_dataloader = dataloaders[1]

    #BERT 모델 불러오기
    model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)

    #optimizer와 schedule 설정
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    t_total = len(train_dataloader) * num_epochs
    warmup_step = int(t_total * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)
    for e in range(num_epochs):
        train_acc = 0.0
        test_acc = 0.0
        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_dataloader)):
            optimizer.zero_grad()
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length= valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            loss = loss_fn(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            train_acc += calc_accuracy(out, label)
            if batch_id % log_interval == 0:
                print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
        print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
        
        model.eval()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length= valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            test_acc += calc_accuracy(out, label)
        print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))
    return model

#정확도 측정을 위한 함수 정의
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

def predict(model, predict_sentence):

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)
    
    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length= valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)


        test_eval=[]
        for i in out:
            logits=i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append(0) # 길안내
            elif np.argmax(logits) == 1:
                test_eval.append(1) # 질문

    return test_eval[0]