import os
import sys
import argparse
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
from models.sweatynet1 import SweatyNet1
from models.sweatynet2 import SweatyNet2
from models.sweatynet3 import SweatyNet3
import utils.load_data as ld
from utils.plot import Plot
from train import train, validate
from test import test_model

parser = argparse.ArgumentParser()
parser.add_argument('--train', default='true', action='store_true', help='choose to train or not (Default: True)')
parser.add_argument('--model' , type = int, default = 3, help = 'select model type - 1 (SweatyNet1) | 2 (SweatyNet2) | 3 (SweatyNet3) (Default: 3)')
# parser.add_argument('--root', default = './data', help = 'root directory for all training and test files')
parser.add_argument('--traincsv', default = './data/imageset_final_annotations.csv', help = 'training set annotations and file names')
parser.add_argument('--trainset', default = './data/imageset_final', help = 'path to training set files')
parser.add_argument('--train_batch_size' , type = int, default = 4, help = 'size of training batch')
parser.add_argument('--testcsv', default = './data/imageset_test_annotations.csv', help = 'test set annotations and file names')
parser.add_argument('--testset', default = './data/imageset_test', help = 'path to test set files')
parser.add_argument('--test_batch_size' , type = int, default = 10, help = 'size of test batch')
parser.add_argument('--checkpoint', default = None, help = 'checkpoint file to load')
parser.add_argument('--workers' , type = int, default = 2, help = 'no of workers to load dataset')
parser.add_argument('--niter' , type = int, default = 25, help = 'no of iterations')
parser.add_argument('--lr' , type = int, default = 0.001, help = 'learning rate')
parser.add_argument('--beta1' , type = int, default = 0.9, help = 'momentum for adam optimizer')
parser.add_argument('--weight_decay' , type = int, default = 0.01, help = 'l2 regularization factor for the optimizer')
parser.add_argument('--outpath' , default = './output', help = 'path to output checkpoint and data')
parser.add_argument('--manual_seed', type=int, default=42, help='manual seed for randomizer')

opt = parser.parse_args()
print(opt)
print('=============================================================')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plotter = Plot()

torch.manual_seed(opt.manual_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(opt.manual_seed)
np.random.seed(opt.manual_seed)

out_checkpoint = os.path.join(opt.outpath, 'checkpoint')
out_media = os.path.join(opt.outpath, 'media')

if os.path.exists(out_checkpoint):
    print(out_checkpoint + ' already exists.')
    print('=============================================================')
else:
    os.makedirs(out_checkpoint)

if os.path.exists(out_media):
    print(out_media + ' already exists.')
    print('=============================================================')
else:
    os.makedirs(out_media)

output_vid = os.path.join(out_media, 'output.mp4')
if os.path.exists(output_vid):
    print(output_vid + ' already exists.')
    bkp_filename = out_media+'/output_bkp_'+datetime.datetime.today().strftime('%Y%m%d%H%M%S')+'.mp4'
    try:
        os.rename(output_vid, bkp_filename)
    except WindowsError:
        os.rename(output_vid, bkp_filename)
        os.remove(output_vid)        
    print('File renamed to ' + bkp_filename)
    print('=============================================================')

if opt.model == 1:
    model_name = 'SweatyNet1'
    model = SweatyNet1(device)
elif opt.model == 2:
    model_name = 'SweatyNet2'
    model = SweatyNet2(device)
elif opt.model == 3:
    model_name = 'SweatyNet3'
    model = SweatyNet3(device)
else:
    sys.exit('Error: Incorrect choice of model. Please select either of 1 (SweatyNet1), 2 (SweatyNet2) or 3 (SweatyNet3)')

print('Model: ' + model_name + ' selected.')
print('=============================================================')

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)

model, criterion = model.to(device), criterion.to(device)

if opt.checkpoint is not None:
    if os.path.exists(opt.checkpoint):
        checkpoint = torch.load(opt.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        min_val_loss = checkpoint['val_loss']
        print('Checkpoint Loaded. Start Epoch: {}, Min Validation Loss: {}'. format(start_epoch, min_val_loss))
        print('=============================================================')
    else:
        sys.exit('Error: Could not load checkpoint file. ' + opt.checkpoint + ' does not exist.')
else:
    start_epoch = 0
    min_val_loss = np.inf
    print('Model Initialized. Start Epoch: {}, Min Validation Loss: {}'. format(start_epoch, min_val_loss))
    print('=============================================================')

if opt.train:
    epochs = opt.niter
else:
    epochs = 1

if opt.train:
    dataset = ld.CustomDataset(opt.traincsv, opt.trainset, transform=[ld.Resize((512, 640)), ld.RandomVerticalFlip(0.5), ld.RandomHorizontalFlip(0.5), ld.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.2), ld.ToTensor()])
    len_trainset = int(0.8*len(dataset))
    len_valset = len(dataset) - len_trainset
    trainset, valset = torch.utils.data.random_split(dataset, [len_trainset, len_valset])

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=opt.train_batch_size, shuffle=True, num_workers=opt.workers)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=opt.test_batch_size, shuffle=False, num_workers=opt.workers)

    win_train_loss = None
    win_val_loss = None
    win_val_recall = None
    win_val_fdr = None
    win_val_conf_mat = None
    win_val_img_grid = None

testset = ld.CustomDataset(opt.testcsv, opt.testset, transform=[ld.Resize((512, 640)), ld.ToTensor()])
test_loader = torch.utils.data.DataLoader(testset, batch_size=opt.test_batch_size, shuffle=False, num_workers=opt.workers)

start_time = time.time()
for epoch in range(start_epoch, epochs):
    if opt.train:
        print('Training in progress...')
        print('-------------------------------------------------------------')
        epoch_start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, epoch, device)
        val_loss, val_confusion_matrix, val_grid = validate(model, val_loader, optimizer, criterion, device)
        if (val_confusion_matrix[1, 1]+val_confusion_matrix[1, 0]) > 0:
            val_recall = val_confusion_matrix[1, 1]/(val_confusion_matrix[1, 1]+val_confusion_matrix[1, 0])
        else:
            val_recall = 0

        if (val_confusion_matrix[1, 1]+val_confusion_matrix[0, 0]) > 0:
            val_fdr = val_confusion_matrix[0, 0]/(val_confusion_matrix[0, 0]+val_confusion_matrix[1, 1])
        else:
            val_fdr = 0

        epoch_end_time = time.time()
        print('Epoch: {}, Train Loss: {}, Validation Loss: {}, Validation Recall: {:.2f}%, Validation FDR: {:.2f}%, Epoch Run Time: {} minute(s) and {} second(s)'.format(epoch+1, train_loss, val_loss, val_recall*100, val_fdr*100, int((epoch_end_time-epoch_start_time)/60), int((epoch_end_time-epoch_start_time)%60)))
        
        win_train_loss = plotter.vis_line(train_loss, epoch, model_name, 'Train Loss', win=win_train_loss)
        win_val_loss = plotter.vis_line(val_loss, epoch, model_name, 'Validation Loss', win=win_val_loss)
        win_val_recall = plotter.vis_line(val_recall*100, epoch, model_name, 'Recall (Validation Set)', win=win_val_recall)
        win_val_fdr = plotter.vis_line(val_fdr*100, epoch, model_name, 'FDR (Validation Set)', win=win_val_fdr)
        win_val_conf_mat = plotter.vis_heatmap(val_confusion_matrix, 'Confusion Matrix (Validation Set-'+model_name+')', win=win_val_conf_mat)
        win_val_img_grid = plotter.vis_image_grids(val_grid, 'Sample Images (Validation Set-'+model_name+')', win=win_val_img_grid)

        if min_val_loss > val_loss:
            torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
            }, out_checkpoint+'/'+model_name+'.pth')
            min_val_loss = val_loss

    if epoch == epochs-1:
        print('=============================================================')
        print('Processing test data...')
        print('-------------------------------------------------------------')
        test_loss, test_confusion_matrix, test_grid = test_model(model, test_loader, optimizer, criterion, device, output_vid=output_vid)
        if (test_confusion_matrix[1, 1]+test_confusion_matrix[1, 0]) > 0:
            test_recall = test_confusion_matrix[1, 1]/(test_confusion_matrix[1, 1]+test_confusion_matrix[1, 0])
        else:
            test_recall = 0

        if (test_confusion_matrix[1, 1]+test_confusion_matrix[0, 0]) > 0:
            test_fdr = test_confusion_matrix[0, 0]/(test_confusion_matrix[0, 0]+test_confusion_matrix[1, 1])
        else:
            test_fdr = 0
        print('Test Loss: {}, Testset Recall: {:.2f}%, Testset FDR: {:.2f}%'.format(test_loss, test_recall*100, test_fdr*100))
        print('=============================================================')
        win_test_conf_mat = plotter.vis_heatmap(test_confusion_matrix, 'Confusion Matrix (Test Set-'+model_name+')', win=None)
        win_test_img_grid = plotter.vis_image_grids(test_grid, 'Sample Images (Test Set-'+model_name+')', win=None)        

end_time = time.time()
hours = int((end_time-start_time)/3600)
minutes = int((end_time-start_time)/60)-(hours*60)
seconds = int((end_time-start_time)%60)
print('Total time taken: {} hour(s), {} minute(s) and {} second(s)'.format(hours, minutes, seconds))
