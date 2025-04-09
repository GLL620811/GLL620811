import torch
from torch.utils.data import DataLoader
from data import BirdAudioDataSet
from utils.AudioUtils import AudioUtils
from model import Resnet18
import numpy as np
from torchvision import transforms
from data import *


def to_feature(dataloader, model, device):
    model.eval()
    output_features_arr = []
    true_total = 0
    all_total = 0
    with torch.no_grad():
        for i, (log_cwt_spec_data, batch_cls_data) in enumerate(dataloader):
            with torch.no_grad():
                batch_cls_data = batch_cls_data.long().to(device)
                input_features = log_cwt_spec_data.float().to(device)
            pred_cls, output_features = model(input_features)
            pred_cls = pred_cls.argmax(-1)
            true_total += ((pred_cls == batch_cls_data).sum().item())
            all_total += batch_cls_data.size(0)
            output_features_arr += torch.cat([output_features, batch_cls_data.unsqueeze(1)],
                                             dim=1).cpu().detach().numpy().tolist()
    print(f'accuracy: {true_total / all_total}')
    return output_features_arr


if __name__ == '__main__':
    batch_size = 128

    train_dir = '/home/dv/Downloads/436773567/New_CWT/train'
    test_dir = '/home/dv/Downloads/436773567/New_CWT/test'
    val_dir = '/home/dv/Downloads/436773567/New_CWT/val'
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.ToTensor()])
    train_bird_dataset = FileFolderDataset(root_dir=train_dir, transform=transform)
    train_loader = DataLoader(dataset=train_bird_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    test_bird_dataset = FileFolderDataset(root_dir=test_dir, transform=transform)
    test_loader = DataLoader(dataset=test_bird_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    val_bird_dataset = FileFolderDataset(root_dir=val_dir, transform=transform)
    val_loader = DataLoader(dataset=val_bird_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = Resnet18(in_channels=1, in_dim=64, num_classes=20).to(device)
    model.load_state_dict(
        torch.load('/home/dv/best_weight/checkpoints-Resnet18-CWT-0.9223334723960506.pth'))
    train_log_spec_feature = to_feature(train_loader, model, device)
    train_log_spec_feature = np.asarray(train_log_spec_feature)
    np.save('../feature/log_cwt_spec/train.npy', train_log_spec_feature)

    test_log_spec_feature = to_feature(test_loader, model, device)
    test_log_spec_feature = np.asarray(test_log_spec_feature)
    np.save('../feature/log_cwt_spec/test.npy', test_log_spec_feature)

    val_log_spec_feature = to_feature(val_loader, model, device)
    val_log_spec_feature = np.asarray(val_log_spec_feature)
    np.save('../feature/log_cwt_spec/val.npy', val_log_spec_feature)

    pass
