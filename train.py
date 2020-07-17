# coding = utf-8
import sys



# Call the subroutine under the folder
sys.path.append('/C-DenseNet/dataset')
sys.path.append('/C-DenseNet/utils')
sys.path.append('/C-DenseNet/models')
import os
import shutil
import argparse
import torch
import torch.optim as optim
from torchvision.models import resnet101
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from utils.train_utils import train,validate,creatdir,test
from dataset.dataset_nocsv import collate_fn, dataset,get_image_pd,defect_label
from dataset.data_aug import *
from models.densenet import DenseNet121
from models.cbamdensenet import cbam_densenet121
from models.cbamdensenet import cbam_densenet169
from models.cbamdensenet import cbam_densenet201
from models.cbamdensenetin import cbam_in_densenet121


torch.backends.cudnn.benchmark = True


# Parameter setting
parser = argparse.ArgumentParser()
# data set path
parser.add_argument('--img_root_train', type=str, default="/C-DenseNet/dataset/train1/train/", help='whether to img root')
# model and data storage path
parser.add_argument('--checkpoint_dir', type=str, default='/C-DenseNet/results/169cbam/', help='directory where model checkpoints are saved')
# Network selection
parser.add_argument('--net', dest='net',type=str, default='DenseNet',help='which net is chosen for training ')
# batch size
parser.add_argument('--batch_size', type=int, default=8, help='size of each image batch')
# learning rate
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
# cuda
parser.add_argument('--cuda', type=str, default="0", help='whether to use cuda if available')
# CPU load data thread setting
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
# Pause settings
parser.add_argument('--resume', type=str, default=None, help='path to resume weights file')
# Number of iterations
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
# Start epoch(set for resume)
parser.add_argument('--start_epoch', type=int, default=0 , help='number of start epoch')
# Interval to display results
parser.add_argument('--print_interval', type=int, default=1, help='interval between print log')
# Verify the parameter and use it in your program through opt.xx
opt = parser.parse_args()
if __name__ == '__main__':
    # Create storage and log files
    creatdir(opt.checkpoint_dir)
    # Get the image path, and divide the training set and validation set
    all_pd=  get_image_pd(opt.img_root_train)
    train_pd,val_pd = train_test_split(all_pd, test_size=0.2, random_state=53,stratify=all_pd["label"])
    # Output data set's size
    print(val_pd.shape)
    # Data augmentation (preprocessing)
    data_transforms = {
        'train1': Compose([
            Resize(size=(640,640)),
            RandomHflip(),
            RandomVflip(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': Compose([
                    Resize(size=(640, 640)),
                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    }

    data_set = {}
    data_set['train1'] = dataset(anno_pd=train_pd, transforms=data_transforms["train1"])
    data_set['val'] = dataset(anno_pd=val_pd, transforms=data_transforms["val"])
    dataloader = {}
    # Load the enhanced training set
    dataloader['train1'] = torch.utils.data.DataLoader(data_set['train1'], batch_size=opt.batch_size,
                                                      shuffle=True, num_workers=opt.n_cpu, collate_fn=collate_fn)
    # Load the enhanced validation set
    dataloader['val'] = torch.utils.data.DataLoader(data_set['val'], batch_size=opt.batch_size,
                                                    shuffle=True, num_workers=opt.n_cpu, collate_fn=collate_fn)
    # Network choosing
    if opt.net == "ResNet":
        model = resnet101(pretrained=True)
        model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        model.fc = torch.nn.Linear(model.fc.in_features, 6)
    elif opt.net == 'DenseNet':
        model = DenseNet121()
    elif opt.net == 'C-DenseNet':
        model = cbam_densenet121()
    elif opt.net == 'C-DenseNet-IN':
        model = cbam_in_densenet121()
    elif opt.net == 'C-DenseNet169':
        model = cbam_densenet169()
    elif opt.net == 'C-DenseNet201':
        model = cbam_densenet201()
    # Pause options
    if opt.resume:
        model.eval()
        print('resuming finetune from %s' % opt.resume)
        try:
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(opt.resume))
        except KeyError:
            model.load_state_dict(torch.load(opt.resume))
            model = torch.nn.DataParallel(model)
    else:
        model = torch.nn.DataParallel(model)
    model = model.cuda()
    # SGD optimization
    optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=1e-4)
    # loss
    criterion = CrossEntropyLoss().cuda()
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    best_precision = 0
    lowest_loss = 10000


    # training
    for epoch in range(opt.start_epoch, opt.epochs):
        # training
        acc_train, loss_train = train(dataloader['train1'], model, criterion, optimizer, exp_lr_scheduler, epoch, print_interval=opt.print_interval,filename=opt.checkpoint_dir)
        # Record the training accuracy and loss of each epoch in the log file
        with open(opt.checkpoint_dir + 'record.txt', 'a') as acc_file:
            acc_file.write('Epoch: %2d, train_Precision: %.8f, train_Loss: %.8f\n' % (epoch, acc_train, loss_train))
        # validate
        precision, avg_loss = validate(dataloader['val'], model, criterion, print_interval=opt.print_interval,filename=opt.checkpoint_dir)
        exp_lr_scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print(epoch, lr)
        # Record the validatory accuracy and loss of each epoch in the log file
        with open(opt.checkpoint_dir + 'record_val.txt', 'a') as acc_file:
            acc_file.write('Epoch: %2d, Precision: %.8f, Loss: %.8f\n' % (epoch, precision, avg_loss))
            # Record the highest accuracy and lowest loss
            is_best = precision > best_precision
            is_lowest_loss = avg_loss < lowest_loss
            best_precision = max(precision, best_precision)
            lowest_loss = min(avg_loss, lowest_loss)
            print('--'*30)
            print(' * Accuray {acc:.3f}'.format(acc=precision), '(Previous Best Acc: %.3f)' % best_precision,
                  ' * Loss {loss:.3f}'.format(loss=avg_loss), 'Previous Lowest Loss: %.3f)' % lowest_loss)
            print('--' * 30)
            # Save the latest model
            save_path = os.path.join(opt.checkpoint_dir,'checkpoint.pth')
            torch.save(model.state_dict(),save_path)
            # Save the model with the highest accuracy
            best_path = os.path.join(opt.checkpoint_dir,'best_model.pth')
            if is_best:
                shutil.copyfile(save_path, best_path)
            # The model with the lowest loss
            lowest_path = os.path.join(opt.checkpoint_dir, 'lowest_loss.pth')
            if is_lowest_loss:
                shutil.copyfile(save_path, lowest_path)

            # # test
            # precision, avg_loss,prec1,rec1,f1,acc1 = test(dataloader['val'], model, criterion, print_interval=opt.print_interval,filename=opt.checkpoint_dir)
            # print(precision, avg_loss,prec1,rec1,f1,acc1)
