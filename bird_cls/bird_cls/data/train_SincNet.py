from data import *
from utils import process_audio_batch, plot_confusion_matrix, process_audio_batch_Morlet
from model import TrUnet, ResNet50, CNN, DwResNet, CNN_1D, MTS, SincNet,MLP
from torch import nn
import visdom
from sklearn.metrics import confusion_matrix
import torchvision
import torch.nn.functional as F

import torch
import torch.nn as nn
from torchvision import transforms

torch.backends.cudnn.enabled = False


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss


def draw_features(features, step):
    x10 = features[9].reshape(-1, 1, 64, 126)
    images_grid = torchvision.utils.make_grid(tensor=x10, nrow=8, padding=10,
                                              normalize=False, value_range=None,
                                              scale_each=False, pad_value=0)
    # summary_writer.add_image(tag='image', img_tensor=images_grid, global_step=step)


def run_one_epoch(epoch_index, model, data_loader, optimizer, criterion, prin_req, optimizer_centloss=None):
    is_train = optimizer is not None
    if is_train:
        for i in model:
            i.train()
    else:
        for i in model:
            i.eval()
    avg_loss = 0
    avg_accuracy = 0
    total = 0
    for i, (batch_waveform_data, batch_lens_data, batch_cls_data) in enumerate(data_loader):
        with torch.no_grad():
            # batch_waveform_data = add_gaussian_noise_torch(batch_waveform_data, snr_db=0)
            batch_cls_data = batch_cls_data.long().cuda()
            # input_features = process_audio_batch_Morlet(batch_waveform_data,wavelet='cmor3-3',level=10,transform=transform).cuda()
            batch_waveform_data = batch_waveform_data.float().cuda()
            # input_features = process_audio_batch(batch_waveforms=batch_waveform_data,nfft=512).cuda()
        sincNet,dnn1_net,dnn2_net = model
        pred_cls = dnn2_net(dnn1_net(sincNet(batch_waveform_data)))
        loss = criterion(pred_cls, batch_cls_data, None)
        if is_train:
            dnn_optimizer, optimizer_dnn1, optimizer_dnn2 = optimizer
            dnn_optimizer.zero_grad()
            optimizer_dnn1.zero_grad()
            optimizer_dnn2.zero_grad()
            loss.backward()
            dnn_optimizer.step()
            optimizer_dnn1.step()
            optimizer_dnn2.step()
        avg_loss += loss.item()
        pri_lab = pred_cls.argmax(-1)
        avg_accuracy += ((pri_lab == batch_cls_data).sum().item())
        total += batch_lens_data.size(0)

        if prin_req != 0 and (i + 1) % prin_req == 0:
            print("epoch %d  %s| step %d | avg loss %.3f | acc %.3f " % (
                epoch_index + 1, 'train' if is_train else 'eval', i + 1, loss.item(),
                avg_accuracy / total))
    return avg_loss / len(data_loader), avg_accuracy / total

# model init param
options = {
    'cnn_N_filt': [80, 60, 60],
    'cnn_len_filt': [251, 5, 5],
    'cnn_max_pool_len': [3, 3, 3],
    'cnn_act': ['leaky_relu', 'leaky_relu', 'leaky_relu'],
    'cnn_drop': [0.3, 0.3, 0.3],
    'cnn_use_laynorm': [True, True, True],
    'cnn_use_batchnorm': [False, False, False],
    'cnn_use_batchnorm_inp': False,
    'cnn_use_laynorm_inp': True,
    'input_dim': 6400,
    'fs': 16000
}
dnn2_options = {
    'input_dim': 2048,
    'fc_lay': [20],
    'fc_drop': [0.3, 0.3,],
    'fc_use_batchnorm': [False],
    'fc_use_laynorm': [False],
    'fc_use_laynorm_inp': True,
    'fc_use_batchnorm_inp': False,
    'fc_act': ['softmax'],
}
dnn1_arch = {'input_dim': 13560,
             'fc_lay': [2048, 2048, 2048],
             'fc_drop': [0.3, 0.3, 0.3],
             'fc_use_batchnorm': [True, True, True],
             'fc_use_laynorm': [False, False, False],
             'fc_use_laynorm_inp': False,
             'fc_use_batchnorm_inp': False,
             'fc_act': ['leaky_relu', 'leaky_relu', 'leaky_relu'],
             }

if __name__ == '__main__':
    epoch_size = 200
    batch_size = 128
    learning_rate = 1e-3
    print_freq = 10
    train_json_file = '/home/dv/Downloads/436773567/BirdsData-16k/train_datas.json'
    test_json_file = '/home/dv/Downloads/436773567/BirdsData-16k/test_datas.json'
    train_bird_dataset = BirdAudioDataSet(json_dir=train_json_file, sample_rate=16000, segment=0.4)
    train_loader = DataLoader(dataset=train_bird_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_bird_dataset = BirdAudioDataSet(json_dir=test_json_file, sample_rate=16000, segment=0.4)
    test_loader = DataLoader(dataset=test_bird_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # model init
    sincNet = SincNet(options).cuda()
    dnn1_net = MLP(dnn1_arch).cuda()
    dnn2_net = MLP(dnn2_options).cuda()
    # model = MTS(in_channels=1,in_dim=64,num_classes=20).cuda()

    optimizer =  torch.optim.RMSprop(sincNet.parameters(), lr=learning_rate, alpha=0.95, eps=1e-8)
    optimizer_dnn1 = torch.optim.RMSprop(dnn1_net.parameters(),lr=learning_rate,alpha=0.95, eps=1e-8)
    optimizer_dnn2 = torch.optim.RMSprop(dnn2_net.parameters(),lr=learning_rate,alpha=0.95, eps=1e-8)

    model = [sincNet,dnn1_net,dnn2_net]

    ce_loss = nn.NLLLoss()
    tmp_loss = 1e+3

    def compute_criterion(pred_cls, batch_cls_data, mult_cls):
        global ce_loss
        x2_loss = ce_loss(pred_cls, batch_cls_data)
        # loss = loss.view(b,c,n)
        return x2_loss


    for epoch in range(epoch_size):
        train_loss, train_aac = run_one_epoch(epoch_index=epoch, model=model, data_loader=train_loader,
                                              optimizer=[optimizer,optimizer_dnn1,optimizer_dnn2], optimizer_centloss=compute_criterion,
                                              criterion=compute_criterion, prin_req=print_freq)
        with torch.no_grad():
            val_loss, val_aac = run_one_epoch(epoch_index=epoch, model=model, data_loader=test_loader, optimizer=None,
                                              criterion=compute_criterion, prin_req=0)
        # vis.line(Y=np.column_stack((train_loss, train_aac,
        #                             val_loss, val_aac)),
        #          X=np.column_stack((epoch, epoch, epoch, epoch)),
        #          win='Loss',
        #          update='append',
        #          opts=dict(markers=False, legend=['Train Loss', 'Train acc',
        #                                           'Test Loss', 'Test acc'])
        #          )
        print(
            f'epoch {epoch + 1} | train avg loss : {train_loss}| train acc : {train_aac}| test avg loss : {val_loss} | test acc : {val_aac}')
        # if tmp_loss > val_loss:
        #     torch.save(model.state_dict(), f'./checkpoints/checkpoints-SincNet-{val_loss}.pth')
        #     tmp_loss = val_loss
