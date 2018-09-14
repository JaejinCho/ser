import numpy as np
import argparse
import torch
import sys
from torch.utils.data import DataLoader
from ser_utils import IEMOCAP_Dataset, my_collate
from ser_utils import CNN, LSTM
from ser_utils import train, val
from ser_utils import save_checkpoint

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Pytorch IEMOCAP')
    # general configuration
    parser.add_argument('--opt', default='adam', type=str,
                        choices=['adam','sgd'],
                        help='Optimizer')
    parser.add_argument('--feats-scp',type=str,help='a path for a feature script (train)')
    parser.add_argument('--feats-scp-val',type=str,help='a path for a feature script (validation)')
    parser.add_argument('--utt2emo',type=str,help='a path for a utt2emo')
    parser.add_argument('--feat-dim',type=int,default=23,help='feature dimension')
    parser.add_argument('--batch-size',type=int,default=40,help='minibatch size')
    parser.add_argument('--val-batch-size',type=int,default=100,help='minibatch size')
    parser.add_argument('--no-shuffle', action='store_true', default=False, help='disables shuffling data in training')
    parser.add_argument('--num-process',type=int,default=4,help='the number of processes')
    parser.add_argument('--gpu', action='store_true', default=False, help='enable gpu training') # if you do --no-cuda, the args set to True, otherwise False
    parser.add_argument('--epochs', type=int, default=100, help='the number of epochs')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status')
    # network architecture
    ## network type
    parser.add_argument('--network', default='cnn', type=str,
                        choices=['cnn','lstm'], help='cnn architecture')
    ## for CNN
    parser.add_argument('--ks', type=int, default=1,
                        help='kernel size in CNN')
    parser.add_argument('--nc', type=int, default=256,
                        help='the number of channels in CNN')
    ## save model
    parser.add_argument('--save-dir', default='./model/', type=str,
                        help='a directory to save models')
    parser.add_argument('--metric', default='loss', type=str,
                        help='a metric to save better models')

    args = parser.parse_args()

    # gpu related setting
    use_gpu = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    ### TODO(JJ): a part to enable deterministic training (i.e. train a exactly same model at every training) - it seems working only in the first epoch (comopared results two trials and they were exactly same in first epoch but from then on, they were different)
    torch.manual_seed(args.seed)
    print("Initialize train dataset ...")
    iemo_dataset_train = IEMOCAP_Dataset(feats_scp_path = args.feats_scp, utt2emo_path = args.utt2emo, feat_dim = args.feat_dim, device_id=device)
    print("Initialize validation dataset ...")
    iemo_dataset_val = IEMOCAP_Dataset(feats_scp_path = args.feats_scp_val, utt2emo_path = args.utt2emo, feat_dim = args.feat_dim, device_id=device)
    train_loader = DataLoader(iemo_dataset_train, batch_size = args.batch_size, shuffle = not args.no_shuffle, num_workers = args.num_process, collate_fn = my_collate)
    val_loader = DataLoader(iemo_dataset_val, batch_size = args.val_batch_size, num_workers = args.num_process, collate_fn = my_collate)

    # Define a model
    if args.network == 'cnn':
        model = CNN(num_channel=args.nc, kernel_size=args.ks).to(device)
    elif args.network == 'lstm':
        ### TODO(JJ): LSTM
        model = LSTM().to(device)
    else:
        print("ERROR: --network is not defined correctly")
        sys.exit(1)
    print(model); print('\n')

    # Setup an optimizer
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters())
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),lr=0.0001)

    # Define dictionaries for dynamically name-changing variables for metrics
    best_metrics = {}
    best_metrics['val_loss'] = np.inf
    best_metrics['val_acc'] = 0
    best_metrics['val_uar'] = 0

    metrics = {}

    # Train a model over epochs
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        metrics['val_loss'], metrics['val_acc'], metrics['val_uar'] = val(args, model, device, val_loader)

        # save best models according to 3 different criterion (acc, uar seem to
        # go along with each other while loss doesn't seem so - from fold1 results)
        if args.metric in ['loss', 'acc', 'uar']: # This line might not be needed but for readability for later metrics to be added
            if args.metric == 'acc' or args.metric == 'uar':
                is_best = metrics['val_{}'.format(args.metric)] > best_metrics['val_{}'.format(args.metric)]
                best_metrics['val_{}'.format(args.metric)] = max(metrics['val_{}'.format(args.metric)],best_metrics['val_{}'.format(args.metric)])
            elif args.metric == 'loss':
                is_best = metrics['val_{}'.format(args.metric)] < best_metrics['val_{}'.format(args.metric)]
                best_metrics['val_{}'.format(args.metric)] = min(metrics['val_{}'.format(args.metric)],best_metrics['val_{}'.format(args.metric)])
            state = {'epoch': epoch, 'state_dict': model.state_dict(),'optimizer': optimizer.state_dict(),
                     'best_val_{}'.format(args.metric): best_metrics['val_{}'.format(args.metric)]}
            # best_val_{}, after a model loaded, will be compared to val_{} in each epoch to save the best model

            save_checkpoint(state, is_best, args) # similar with case 2 in https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorcq
            # ??? is the model (also optimizer) given to train as reference? meaning can
            # model.state_dict() save a model where its weights is updated in
            # the epoch?
        else:
            print("ERROR: Set metric as one of these: loss, acc, uar")
            sys.exit(1)

if __name__ == '__main__':
    main()
