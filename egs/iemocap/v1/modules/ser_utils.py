import sys
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from struct import unpack
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.utils.data import Dataset


class CNN(nn.Module):
    def __init__(self, dim_feat=23, num_class=4, num_channel=256, kernel_size=1):
        super(CNN,self).__init__()
        self.dim_feat = dim_feat
        self.num_class = num_class
        self.cnn = nn.Conv1d(dim_feat, num_channel, kernel_size, stride=1)
        self.fc1 = nn.Linear(num_channel, num_channel) # *** Try not to use it later
        self.fc2 = nn.Linear(num_channel ,num_class)

    def batchpool(self,x):
        '''
        Currently, batchpool is used as global function. (so this ft is not used here. Will see which way is better)
        !!! Important !!!: Assuming batch_first = True. i.e the dim of x is (batch_size, seq_len, feature_dim)
        - x: feature (PackedSequence)
        '''
        pool_x = [] # dim of each element in pool_x list will be 1
        x = pad_packed_sequence(x, batch_first=True) # x is tuple composed of x[0] as padded sequences and x[1] as lengths
        for sample, length in zip(x[0],x[1]):
            pool_x.append(torch.mean(sample[0:length],dim=0))
        return torch.stack(pool_x,dim=0) # dimensionality of this returning variable == 2

    def forward(self,x,args):
        '''
        0. sorting in decreasing order in length is done during the train loop
        1. x (list) is given as a batch of variable length sequences
        '''
        x = pad_sequence(x, batch_first=True).transpose(1,2)
        if args.gpu and torch.cuda.is_available(): x.cuda()
        x = self.cnn(x)
        x = torch.mean(x,dim=2) # pooling a long the time axis
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training) # https://github.com/pytorch/examples/blob/master/mnist/main.py, model.train() or model.eval() affects the bool value in self.training here (model.training in general)
        x = self.fc2(x)
        return x

### TODO(JJ): change to LSTM
class LSTM(nn.Module):
    def __init__(self, dim_feat=23, num_class=4, num_layer=2, num_channel=256):
        super(LSTM,self).__init__()
        self.dim_feat = dim_feat
        self.num_class = num_class
        self.num_layer = num_layer
        self.lstm = nn.LSTM(self.dim_feat, num_channel, num_layers=self.num_layer, batch_first=True)
        self.fc1 = nn.Linear(num_channel, num_channel)
        self.fc2 = nn.Linear(num_channel, self.num_class)

    def batchpool(self,x):
        '''
        Currently, batchpool is used as global function. (so this ft is not used here. Will see which way is better)
        !!! Important !!!: Assuming batch_first = True. i.e the dim of x is (batch_size, seq_len, feature_dim)
        - x: feature (PackedSequence)
        '''
        pool_x = [] # dim of each element in pool_x list will be 1
        x = pad_packed_sequence(x, batch_first=True) # x is tuple composed of x[0] as padded sequences and x[1] as lengths
        for sample, length in zip(x[0],x[1]):
            pool_x.append(torch.mean(sample[0:length],dim=0))
        return torch.stack(pool_x,dim=0) # dimensionality of this returning variable == 2

    def forward(self,x,args):
        '''
        0. sorting in decreasing order in length is done during the train loop
        1. x (list) is given as a batch of variable length sequences
        '''
        x = pack_sequence(x)
        if args.gpu and torch.cuda.is_available(): x.cuda()
        x,h = self.lstm(x)
        x = batchpool(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class IEMOCAP_Dataset(Dataset): # Instance of this class object will be used with the Dataloader class when instantiatied
    '''
    For detailed implementation, refer to the official example at https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    '''
    def __init__(self, feats_scp_path, utt2emo_path, feat_dim, transform=None, list_emo = ['ang','hap','neu','sad'], device_id="cpu"):
        '''
        input arguments:
        list_emo: subset from a set of all emotion classes to be used for an experiment
        transform: it is not being used for now (it is actually used for Dataset class)
        '''
        self.feats_scp = open(feats_scp_path).readlines()
        self.dict_lab2num = {emo: ix for ix, emo in enumerate(list_emo)} # !!!: may need to be changed
        self.dict_utt2emo = self.create_dict(utt2emo_path, self.feats_scp, list_emo)
        self.feat_dim = feat_dim
        self.transform = transform # NOT BEING USED NOW
        self.device_id = device_id
        print("n_samples per class:\n")
        labels = np.array(list(self.dict_utt2emo.values()))
        for emo in list_emo:
            n_sample = sum(labels == self.dict_lab2num[emo])
            print('{}: {}'.format(emo, n_sample))
        print('\n')

    def create_dict(self, utt2emo_path, feats_scp, list_emo):
        '''
        output: dict mappling from uttid to the label ix
        '''
        uttlist = [ line.split()[0] for line in feats_scp ]
        dict_utt2emo = {}
        for line in open(utt2emo_path):
            uttid, lab = line.strip().split()
            if (uttid in uttlist) and (lab in list_emo):
                dict_utt2emo[uttid] = int(self.dict_lab2num[lab])
        return dict_utt2emo

    def feat_from_ark(self, scp_line):
        '''
        output: a feat seq., the label ix
        '''
        uttid, pos = scp_line.strip().split()
        ark_path, offset = pos.split(':')
        offset = int(offset)
        fin = open(ark_path,'rb')
        fin.seek(offset+6)
        seq_len = unpack('i',fin.read(4))[0]
        fin.seek(offset+15)
        feat = np.fromstring(fin.read(seq_len*self.feat_dim*4), dtype=np.float32).reshape(seq_len, self.feat_dim)
        return torch.from_numpy(feat).to(self.device_id), torch.from_numpy(np.array(self.dict_utt2emo[uttid]))

    def __len__(self):
        return len(self.feats_scp)

    def __getitem__(self,idx):
        return self.feat_from_ark(self.feats_scp[idx])

def batchpool(x):
    '''
    This pools features along seq_len axis while seq_len is differnt by example
    !!! Important !!!: Assuming batch_first = True. i.e the dim of x is (batch_size, seq_len, feature_dim)
    - x: feature (PackedSequence)
    '''
    pool_x = [] # dim of each element in pool_x list will be 1
    x = pad_packed_sequence(x, batch_first=True) # x is tuple composed of x[0] as padded sequences and x[1] as lengths
    for sample, length in zip(x[0],x[1]):
        pool_x.append(torch.mean(sample[0:length],dim=0))
    return torch.stack(pool_x,dim=0) # dimensionality of this returning variable == 2

def my_collate(batch):
    '''
    This enables to make a batch from seq. having diff. len.
    '''
    # sorting batch(a list of (feat, lab) tuples)
    batch = sorted(batch, key=lambda tup: tup[0].size(0),reverse=True)
    b_feat = [ item[0] for item in batch ]
    b_label = [ item[1] for item in batch ]
    return b_feat, torch.tensor(b_label) # torch.Tensor is an alias for torch.FloatTensor (checked w/ v0.4.0)
    #return b_feat, torch.from_numpy(np.array(b_label)) # use this if the above has an error when checking

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    n_example = len(train_loader.dataset)
    for batch_ix, (feat_seq, label) in enumerate(train_loader):
        #feat_seq, label = feat_seq.to(device), label.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(feat_seq, args)
        loss = F.cross_entropy(output,label)
        loss.backward()
        optimizer.step()
        if batch_ix % args.log_interval == 0:
            if len(feat_seq) != args.batch_size:
                n_sample_processed = batch_ix * args.batch_size + len(feat_seq) # should be the number of whole examples
                if n_sample_processed != len(train_loader.dataset):
                    print("ERROR: The number of samples processed in one epoch does NOT match with the number of shole samples")
                    sys.exit(1)
            else:
                n_sample_processed = (batch_ix+1) * args.batch_size # +1 is for one mini-batch you processed in the beginning

            # Print sample-level average loss (i.e. loss averaged over samples in a mini-batch)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, n_sample_processed, n_example,
                100. * batch_ix / len(train_loader), loss.item()))

def val(args, model, device, val_loader):
    model.eval()
    n_example = len(val_loader.dataset)
    val_loss = 0
    correct = 0
    with torch.no_grad():
        label_all = []
        pred_all = []

        for feat_seq, label in val_loader:
            #feat_seq, label = feat_seq.to(device), label.to(device)
            label = label.to(device)
            output = model(feat_seq, args)
            val_loss += F.cross_entropy(output, label)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()
            label_all.append(label)
            pred_all.append(pred.view_as(label))

    label_all = torch.cat(label_all)
    pred_all = torch.cat(pred_all)
    dict_class2acc = acc_perclass(pred_all, label_all, val_loader.dataset.dict_lab2num)

    val_loss /= n_example # sample-level average loss (i.e. loss averaged over samples in a mini-batch)
    val_acc = 100. * correct / n_example
    val_uar = np.mean(list(dict_class2acc.values()))
    print('\nval set:\n Average loss: {:.4f}\tAccuracy: {}/{} ({:.2f}%)\tUAR: {:.2f}%\n Accuracy per class: {}\n'.format(
        val_loss, correct, n_example, val_acc, val_uar, dict_class2acc))

    return val_loss, val_acc, val_uar

def acc_perclass(pred,label,dict_lab2num):
    '''
    Calculate UAR
    '''
    dict_lab2acc = {}
    for lab in dict_lab2num:
        ix = (label == dict_lab2num[lab])
        acc = float(torch.mean((pred[ix] == label[ix]).float()) * 100)
        dict_lab2acc[lab] = acc
    uar = np.mean(list(dict_lab2acc.values()))
    return dict_lab2acc

def save_checkpoint(state, is_best, args):
    save_path = args.save_dir + '/' + args.metric + '_' + str(state['epoch']) + 'epoch_' + args.network + '_' + args.opt + '_' + 'checkpoint.tar'
    torch.save(state, save_path)
    if is_best:
        print("Saving best model...")
        shutil.copyfile(save_path, args.save_dir + args.metric + '_' + 'model_best.tar')
