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
import imageio
from models.sweatynet1 import SweatyNet1
from models.sweatynet2 import SweatyNet2
from models.sweatynet3 import SweatyNet3
import utils.load_data as ld
from utils.plot import Plot
from train import train, __freeze_SweatyNet__
from test import test, detect_ball

parser = argparse.ArgumentParser()
parser.add_argument('--train_off', default=False, action='store_true',
                    help='Turn off training (Default: False)')
parser.add_argument('--model', type=int, default=3,
                    help='select model type - 1 (SweatyNet1) | 2 (SweatyNet2) | 3 (SweatyNet3) (Default: 3)')
parser.add_argument('--traincsv', default='./data/imageset_train_annotations.csv',
                    help='training set annotations and file names')
parser.add_argument('--trainset', default='./data/train',
                    help='path to training set files')
parser.add_argument('--train_batch_size', type=int,
                    default=4, help='size of training batch')
parser.add_argument('--testcsv', default='./data/imageset_test_annotations.csv',
                    help='test set annotations and file names')
parser.add_argument('--testset', default='./data/test',
                    help='path to test set files')
parser.add_argument('--test_batch_size', type=int,
                    default=10, help='size of test batch')
parser.add_argument('--checkpoint', default=None,
                    help='checkpoint file to load')
parser.add_argument('--workers', type=int, default=2,
                    help='no of workers to load dataset')
parser.add_argument('--niter', type=int, default=25, help='no of iterations')
parser.add_argument('--lr', type=int, default=0.001, help='learning rate')
parser.add_argument('--beta1', type=int, default=0.9,
                    help='momentum for adam optimizer')
parser.add_argument('--outpath', default='./output',
                    help='path to output checkpoint and data')
parser.add_argument('--manual_seed', type=int, default=42,
                    help='manual seed for randomizer')
parser.add_argument('--freeze_wts', default=False, action='store_true',
                    help='Freeze weights of SweatyNet (Default: False)')
parser.add_argument('--use_ConvLSTM', default=False, action='store_true',
                    help='Add ConvLSTM to model (Default: False)')
parser.add_argument('--image_height', type=int, default=512, 
                    help='the height of the input image to network')
parser.add_argument('--image_width', type=int, default=640, 
                    help='the width of the input image to network')
parser.add_argument('--input_vid', default=None,
                    help='Input video file to process. Training will be turned off.')
                    

opt = parser.parse_args()
print(opt)
print('=============================================================')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Object for class with visualization functions
plotter = Plot()

torch.manual_seed(opt.manual_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(opt.manual_seed)
np.random.seed(opt.manual_seed)

# Create reader to process input video if provided
# Training will be turned off in this case
if opt.input_vid is not None:
    if not os.path.exists(opt.input_vid):
        sys.exit('Error: '+opt.input_vid+' file does not exist.')
    reader = imageio.get_reader(opt.input_vid)
    opt.image_width, opt.image_height = reader.get_meta_data()['size']
    print('Video reader created. Frame Size: ({}, {})'.format(opt.image_height, opt.image_width))
    print('=============================================================')
    if not opt.train_off:
        opt.train_off = True

# Set paths for checkpoint and media
out_checkpoint = os.path.join(opt.outpath, 'checkpoint')
out_media = os.path.join(opt.outpath, 'media')

# Create checkpoint directory
if os.path.exists(out_checkpoint):
    print(out_checkpoint + ' already exists.')
    print('=============================================================')
else:
    os.makedirs(out_checkpoint)

# Create media directory
if os.path.exists(out_media):
    print(out_media + ' already exists.')
    print('=============================================================')
else:
    os.makedirs(out_media)

# Backup output video file if it already exists to prevent overwriting
output_vid = os.path.join(out_media, 'output.mp4')
if os.path.exists(output_vid):
    print(output_vid + ' already exists.')
    bkp_filename = out_media+'/output_bkp_' + \
        datetime.datetime.today().strftime('%Y%m%d%H%M%S')+'.mp4'
    try:
        os.rename(output_vid, bkp_filename)
    except WindowsError:
        os.rename(output_vid, bkp_filename)
        os.remove(output_vid)
    print('File renamed to ' + bkp_filename)
    print('=============================================================')

# Select model
if opt.model == 1:
    model_name = 'SweatyNet1'
    model = SweatyNet1((opt.image_height, opt.image_width, 3), device, use_ConvLSTM=opt.use_ConvLSTM, seq_len=2)
elif opt.model == 2:
    model_name = 'SweatyNet2'
    model = SweatyNet2((opt.image_height, opt.image_width, 3), device, use_ConvLSTM=opt.use_ConvLSTM, seq_len=4)
elif opt.model == 3:
    model_name = 'SweatyNet3'
    model = SweatyNet3((opt.image_height, opt.image_width, 3), device, use_ConvLSTM=opt.use_ConvLSTM, seq_len=2)
else:
    sys.exit('Error: Incorrect choice of model. Please select either of 1 (SweatyNet1), 2 (SweatyNet2) or 3 (SweatyNet3)')

if opt.use_ConvLSTM:
    model_name += '_ConvLSTM'

print('Model: ' + model_name + ' selected.')
print('=============================================================')

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

model, criterion = model.to(device), criterion.to(device)

# Load checkpoint
if opt.checkpoint is not None:
    if os.path.exists(opt.checkpoint):
        checkpoint = torch.load(opt.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Checkpoint {} loaded.'. format(opt.checkpoint))
        print('=============================================================')
    else:
        sys.exit('Error: Could not load checkpoint file. ' +
                 opt.checkpoint + ' does not exist.')
else:    
    print('Model Initialized.')
    print('=============================================================')

# Freeze weights of the SweatyNet part of the model
if opt.freeze_wts:
    __freeze_SweatyNet__(model, requires_grad=False)
    print('SweatyNet Weights Frozen')
    print('=============================================================')

# Load the datasets
if not opt.train_off:
    epochs = opt.niter
    min_val_loss = np.inf
    if opt.freeze_wts and opt.use_ConvLSTM:
        # Freeze weights whenever Conv LSTM is trained as otherwise the output of
        # SweatyNet sections might not be proper. Shuffling also needs to be turned off.
        transforms = [ld.Resize((opt.image_height, opt.image_width)), ld.ToTensor()]
        dataset = ld.CustomDataset(opt.traincsv, opt.trainset, transform=transforms)
        len_trainset = int(0.8*len(dataset))
        train_indices = torch.arange(200, 200+len_trainset)
        val_indices = torch.cat((torch.arange(0, 200), torch.arange(200+len_trainset, len(dataset))))
        trainset = ld.Subset(dataset, train_indices)
        valset = ld.Subset(dataset, val_indices)
        train_shuffle = False
    else:
        # For normal training, randomly set the training and validation set. Color Jitter
        # and shuffling also improves the rate of convergence and generalization capability
        transforms = [ld.Resize((opt.image_height, opt.image_width)), ld.RandomHorizontalFlip(
            0.5), ld.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), ld.ToTensor()]
        dataset = ld.CustomDataset(opt.traincsv, opt.trainset, transform=transforms)
        len_trainset = int(0.8*len(dataset))
        len_valset = len(dataset) - len_trainset
        trainset, valset = ld.random_split(dataset, [len_trainset, len_valset])
        train_shuffle = True
    
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=opt.train_batch_size, shuffle=train_shuffle, num_workers=opt.workers)
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=opt.test_batch_size, shuffle=False, num_workers=opt.workers)

    # Set the visdom windows to plot metrics
    win_train_loss = None
    win_val_loss = None
    win_val_recall = None
    win_val_fdr = None
    win_val_conf_mat = None
    win_val_img_grid = None

    print('Training in progress...')
    print('-------------------------------------------------------------')
else:
    # To test the model, just iterate over 1 epoch.
    epochs = 1

# Separate testset is not necessary while processing a video
if opt.input_vid is None:
    testset = ld.CustomDataset(opt.testcsv, opt.testset, transform=[
                            ld.Resize((opt.image_height, opt.image_width)), ld.ToTensor()])
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=opt.test_batch_size, shuffle=False, num_workers=opt.workers)

# Iterate over training(optional) and testing
start_time = time.time()
for epoch in range(epochs):
    if not opt.train_off:
        
        epoch_start_time = time.time()
        train_loss = train(model, train_loader, optimizer,
                           criterion, epoch, device, log_interval=225)
        val_loss, val_confusion_matrix, val_grid = test(
            model, val_loader, criterion, device, writer=None)
        # Calculate validation recall
        if (val_confusion_matrix[1, 1]+val_confusion_matrix[1, 0]) > 0:
            val_recall = val_confusion_matrix[1, 1] / \
                (val_confusion_matrix[1, 1]+val_confusion_matrix[1, 0])
        else:
            val_recall = 0

        # Calculate validation false detection rate
        if (val_confusion_matrix[1, 1]+val_confusion_matrix[0, 1]) > 0:
            val_fdr = val_confusion_matrix[0, 1] / \
                (val_confusion_matrix[0, 1]+val_confusion_matrix[1, 1])
        else:
            val_fdr = 0

        epoch_end_time = time.time()
        print('Epoch: {}, Train Loss: {}, Validation Loss: {}, Validation Recall: {:.2f}%, Validation FDR: {:.2f}%, Epoch Run Time: {} minute(s) and {} second(s)'.format(
            epoch+1, train_loss, val_loss, val_recall*100, val_fdr*100, int((epoch_end_time-epoch_start_time)/60), int((epoch_end_time-epoch_start_time) % 60)))

        # Plot metrics on training data
        win_train_loss = plotter.vis_line(
            train_loss, epoch, model_name, 'Train Loss', win=win_train_loss)
        win_val_loss = plotter.vis_line(
            val_loss, epoch, model_name, 'Validation Loss', win=win_val_loss)
        win_val_recall = plotter.vis_line(
            val_recall*100, epoch, model_name, 'Recall (Validation Set)', win=win_val_recall)
        win_val_fdr = plotter.vis_line(
            val_fdr*100, epoch, model_name, 'FDR (Validation Set)', win=win_val_fdr)
        win_val_conf_mat = plotter.vis_heatmap(
            val_confusion_matrix, 'Confusion Matrix (Validation Set-'+model_name+')', win=win_val_conf_mat)
        win_val_img_grid = plotter.vis_image_grids(
            val_grid, 'Sample Images (Validation Set-'+model_name+')', nrows=3, win=win_val_img_grid)

        if min_val_loss > val_loss:
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, out_checkpoint+'/'+model_name+'.pth')
            min_val_loss = val_loss
    else:
        print('Training skipped.')

    if epoch == epochs-1:
        # Process either one of testset data or an input video.
        if opt.input_vid is None:
            print('=============================================================')
            print('Processing test data...')
            print('-------------------------------------------------------------')
            writer = imageio.get_writer(output_vid, fps=6)
            test_loss, test_confusion_matrix, test_grid = test(
                model, test_loader, criterion, device, writer=writer)
            writer.close()
            
            # Calculate testset recall
            if (test_confusion_matrix[1, 1]+test_confusion_matrix[1, 0]) > 0:
                test_recall = test_confusion_matrix[1, 1]/(
                    test_confusion_matrix[1, 1]+test_confusion_matrix[1, 0])
            else:
                test_recall = 0

            # Calculate testset false detection rate
            if (test_confusion_matrix[1, 1]+test_confusion_matrix[0, 1]) > 0:
                test_fdr = test_confusion_matrix[0, 1] / \
                    (test_confusion_matrix[0, 1]+test_confusion_matrix[1, 1])
            else:
                test_fdr = 0

            # Plot metrics on test data
            print('Test Loss: {}, Testset Recall: {:.2f}%, Testset FDR: {:.2f}%'.format(
                test_loss, test_recall*100, test_fdr*100))
            print('=============================================================')
            plotter.vis_heatmap(
                test_confusion_matrix, 'Confusion Matrix (Test Set-'+model_name+')', win=None)
            plotter.vis_image_grids(
                test_grid, 'Sample Images (Test Set-'+model_name+')', nrows=3, win=None)
        else:
            print('=============================================================')
            print('Processing video file...')
            print('-------------------------------------------------------------')
            detect_ball(model, reader, output_vid, device)
            reader.close()
            print('File processed and output saved to {}'.format(output_vid))
            print('=============================================================')

end_time = time.time()
hours = int((end_time-start_time)/3600)
minutes = int((end_time-start_time)/60)-(hours*60)
seconds = int((end_time-start_time) % 60)
print('Total time taken: {} hour(s), {} minute(s) and {} second(s)'.format(
    hours, minutes, seconds))