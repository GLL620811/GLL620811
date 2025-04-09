from data import *
from utils import process_audio_batch, plot_confusion_matrix, process_audio_batch_Morlet,add_gaussian_noise_torch
from model import TrUnet, ResNet50, CNN, DwResNet, CNN_1D, MTS, SincNet, NewSincNet
import torch.nn.functional as F
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset

def caculate_ACI(input_features):
    # d_k_t = np.abs(torch.diff(input_features, axis=2))
    # d_t = np.sum(d_k_t, axis=0)
    # aci_t = d_t / input_features.sum(0)
    # aci = aci_t.mean()
    d_k_t = torch.abs(torch.diff(input_features, axis=2))
    d_t = torch.sum(d_k_t, axis=2)
    aci_t = d_t / (input_features.sum(2) + 1e-8)
    aci = aci_t.mean(-1)
    return aci


if __name__ == '__main__':
    epoch_size = 50
    batch_size = 128
    learning_rate = 1e-3
    print_freq = 10
    train_json_file = '/home/dv/Downloads/436773567/BirdsData-16k/train_datas.json'
    test_json_file = '/home/dv/Downloads/436773567/BirdsData-16k/test_datas.json'
    val_json_file = '/home/dv/Downloads/436773567/BirdsData-16k/val_datas.json'

    val_bird_dataset = BirdAudioDataSet(json_dir=val_json_file, sample_rate=16000, segment=0.4)
    train_bird_dataset = BirdAudioDataSet(json_dir=train_json_file, sample_rate=16000, segment=0.4)
    combined_dataset = ConcatDataset([val_bird_dataset, train_bird_dataset])
    train_loader = DataLoader(dataset=combined_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    test_bird_dataset = BirdAudioDataSet(json_dir=test_json_file, sample_rate=16000, segment=0.4)
    test_loader = DataLoader(dataset=test_bird_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = MTS(in_channels=1, in_dim=64, num_classes=20).cuda()
    model.load_state_dict(torch.load('/home/dv/best_mts_weight/checkpoints-MTS-Weight-LogSpec-0.9543874287303574.pth'))
    # model = Resnet18(in_channels=1, in_dim=64, num_classes=20).cuda()
    for i, (batch_waveform_data, batch_lens_data, batch_cls_data) in enumerate(train_loader):
        with torch.no_grad():
            batch_cls_data = batch_cls_data.long().cuda()
            input_features = process_audio_batch(batch_waveforms=batch_waveform_data, nfft=512).cuda()
            old_aci = caculate_ACI(input_features)
            pred_cls, muti_features,output_features = model(torch.log10(input_features + 1e-8).unsqueeze(1))
            new_aci = caculate_ACI(output_features)

            # 绘制折线图
            plt.figure(figsize=(10, 5))
            plt.plot(old_aci.squeeze().cpu().detach().numpy(), label='Old ACI', marker='o')
            plt.plot(F.sigmoid(new_aci).mean(-1).cpu().detach().numpy(), label='New ACI', marker='o')
            plt.title('ACI Comparison')
            plt.xlabel('Sample Index')
            plt.ylabel('ACI Value')
            plt.legend()
            plt.grid(True)
            plt.show()
            print(old_aci.squeeze().cpu().detach().numpy().mean())
            print(old_aci.squeeze().cpu().detach().numpy().std())
            print(new_aci.mean(-1).cpu().detach().numpy().mean())
            print(new_aci.mean(-1).cpu().detach().numpy().std())
            break
        pass