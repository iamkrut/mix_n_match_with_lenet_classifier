import sys
import os
from os.path import join
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from torchvision import transforms

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


from model import My_LeNET
from dataloader import ChairsDataset
from torch.utils import data

from data_preprocess import preproc, preproc_2

from PIL import Image


def train_net(net, data_dir):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    cudnn.benchmark = True

    params = {'batch_size': 100,
            'shuffle': True,
            'num_workers': 3}
    max_epochs = 10


    train_dataset = ChairsDataset(root_dir='chairs-data', train=True,
                                  transform=transforms.Compose( [
                                  transforms.Resize( ( 56, 56 ) ),
                                  transforms.ToTensor(),
                                  transforms.Normalize( ( 0.5, 0.5, 0.5 ),
                                                        ( 0.5, 0.5, 0.5 ) ) 
                                                        ] ) )
    training_generator = data.DataLoader(train_dataset, **params)

    test_dataset = ChairsDataset(root_dir='chairs-data', train=False,
                                  transform=transforms.Compose( [
                                  transforms.Resize( ( 56, 56 ) ),
                                  transforms.ToTensor(),
                                  transforms.Normalize( ( 0.5, 0.5, 0.5 ),
                                                        ( 0.5, 0.5, 0.5 ) ) 
                                                        ] ) )
    testing_generator = data.DataLoader(test_dataset, **params)

    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    weights = [1/train_dataset.negative_size(), 1/train_dataset.positive_size()]
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    net = net.to(device)

    for epoch in range(max_epochs):


        print('Epoch %d/%d' % (epoch + 1, max_epochs))
        print('Training...')
        
        net.train()
        epoch_loss = 0


        for i, (local_batch, local_labels) in enumerate(training_generator):

            local_batch, local_labels = local_batch.float().to(device), local_labels.to(device)


            optimizer.zero_grad()
            pred = net.forward(local_batch)
            loss = criterion(pred, local_labels)

            epoch_loss += loss.detach().cpu().item()

            print('Training batch %d  - Loss: %.6f' % (i+1, loss.detach().cpu().item()))
        

            loss.backward()
            optimizer.step()
            

        torch.save(net.state_dict(), join(data_dir, 'checkpoints') + '/CP%d.pth' % (epoch + 1))
        print('Checkpoint %d saved !' % (epoch + 1))
        print('Epoch %d finished! - Loss: %.6f' % (epoch+1, epoch_loss / i))

    precision = []
    recall = []
    net.eval()
    with torch.no_grad():
        for i, (local_batch, local_labels) in enumerate(testing_generator):

            local_batch = local_batch.float().to(device)

            pred = net.forward(local_batch)

            temp = torch.zeros((pred.shape[0]))
            for i in range(pred.shape[0]):
                if pred[i,0]<pred[i,1] :
                    temp[i] = 1

            gt = local_labels.cpu().detach().numpy()
            pd = temp.cpu().detach().numpy()
            pd = np.int64(pd)

            precision.append(precision_score(gt, pd))
            recall.append(recall_score(gt, pd))

    f = open("results.txt","w+")
    f.write('presicion : {}\r\n'.format(np.mean(precision)))
    f.write('recall : {}\r\n'.format(np.mean(recall)))
    f.close()

def just_test(net, data_dir='evaluate-chairs'):

    print("just testing!")

    files = os.listdir(data_dir)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    cudnn.benchmark = True

    net = net.to(device)

    net.eval()

    for f in files :
        print(f)
        image = Image.open(join(data_dir,f))
        image = transforms.functional.resize(image,(56,56))
        image = transforms.functional.to_tensor(image)
        image = transforms.functional.normalize(image,(0.5,0.5,0.5),(0.5,0.5,0.5))
        image = torch.unsqueeze(image, dim=0)

        image = image.to(device)
        pred = net.forward(image)

        temp = torch.zeros((pred.shape[0]))
        for i in range(pred.shape[0]):
            if pred[i,0]<pred[i,1] :
                temp[i] = 1

        print(temp)

def test_concat(data_dir='evaluate-chairs'):
    for filename in os.listdir(data_dir):
            
            
            view = int(filename.split(".")[0]) 
            idx = view
            view = view % 3

            if view == 1:

                filename_1 = join(data_dir,filename)
                filename_2 = join(data_dir, '{}.bmp'.format(str(idx+1)))
                filename_3 = join(data_dir, '{}.bmp'.format(str(idx+2)))
                try :
                    image_1 = Image.open(filename_1)
                except:
                    print("no such file : ",filename_1)
                    continue
                try:
                    image_2 = Image.open(filename_2)
                except:
                    print("no such file : ",filename_2)
                    continue
                try:
                    image_3 = Image.open(filename_3)
                except:
                    print("no such file : ",filename_3)
                    continue
                image_1 = np.array(image_1)
                image_2 = np.array(image_2)
                image_3 = np.array(image_3)
                image = np.concatenate((image_1,image_2,image_3), axis=1)
                image = Image.fromarray(image)
                image.save(filename_1)
                os.remove(filename_2)
                os.remove(filename_3)




def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=10, type='int', help='number of epochs')
    parser.add_option('-d', '--data-dir', dest='data_dir', default='chairs-data', help='data directory')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default=False, help='use cuda')
    parser.add_option('-l', '--load', dest='load', default=False, help='load file model')
    parser.add_option('-p', '--pre_process', action='store_true', default=False, help='need_pre_process?')
    parser.add_option('-c', '--concatination', action='store_true', default=False, help='need_concatination?')
    parser.add_option('-t', '--just_test', action='store_true', default=False, help='just testing!')
    parser.add_option('-a', '--test_concat', action='store_true', default=False, help='test files concat')



    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()

    if args.concatination:
        preproc(root_dir=args.data_dir)

    if args.pre_process:
        preproc_2(root_dir=args.data_dir)

    

    net = My_LeNET()

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from %s' % (args.load))

    if args.test_concat:
        test_concat()

    if args.just_test :
        just_test(net)
        exit(0)

    
    train_net(net, data_dir=args.data_dir)
