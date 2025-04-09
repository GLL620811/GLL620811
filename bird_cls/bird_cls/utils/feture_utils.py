import torch
from torch.utils.data import DataLoader
from data import BirdAudioDataSet
from utils.AudioUtils import AudioUtils
from model import Resnet18
import numpy as np
def to_feature(dataloader, model, device):
    audio_utils = AudioUtils()
    model.eval()
    output_features_arr = []
    true_total = 0
    all_total = 0
    with torch.no_grad():
        for i, (batch_waveform_data, batch_lens_data, batch_cls_data) in enumerate(dataloader):
            with torch.no_grad():
                batch_cls_data = batch_cls_data.long().to(device)
                input_features = audio_utils.process_audio_batch_ToLogMelSpec(batch_waveform_data, nfft=512,n_mels=40).to(device)
            pred_cls,output_features = model(input_features)
            pred_cls = pred_cls.argmax(-1)
            true_total += ((pred_cls == batch_cls_data).sum().item())
            all_total += batch_lens_data.size(0)
            output_features_arr += torch.cat([output_features,batch_cls_data.unsqueeze(1)],dim=1).cpu().detach().numpy().tolist()
    print(f'accuracy: {true_total/all_total}')
    return output_features_arr


if __name__ == '__main__':
    batch_size = 128
    train_json_file = '/home/dv/Downloads/436773567/BirdsData-16k/train_datas.json'
    test_json_file = '/home/dv/Downloads/436773567/BirdsData-16k/test_datas.json'
    val_json_file = '/home/dv/Downloads/436773567/BirdsData-16k/val_datas.json'
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    train_bird_dataset = BirdAudioDataSet(json_dir=train_json_file, sample_rate=16000, segment=0.4)
    train_loader = DataLoader(dataset=train_bird_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    test_bird_dataset = BirdAudioDataSet(json_dir=test_json_file, sample_rate=16000, segment=0.4)
    test_loader = DataLoader(dataset=test_bird_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    val_bird_dataset = BirdAudioDataSet(json_dir=val_json_file,sample_rate=16000,segment=0.4)
    val_loader = DataLoader(dataset=val_bird_dataset, batch_size=batch_size,shuffle=False)
    model = Resnet18(in_channels=1, in_dim=64, num_classes=20).to(device)
    model.load_state_dict(torch.load('/home/dv/best_weight/checkpoints-Resnet18-LogMel-0.21968758973268282.pth'))
    train_log_spec_feature = to_feature(train_loader,model,device)
    train_log_spec_feature = np.asarray(train_log_spec_feature)
    np.save('../feature/log_mel_spec/train.npy',train_log_spec_feature)

    test_log_spec_feature = to_feature(test_loader, model, device)
    test_log_spec_feature = np.asarray(test_log_spec_feature)
    np.save('../feature/log_mel_spec/test.npy',test_log_spec_feature)

    val_log_spec_feature = to_feature(val_loader, model, device)
    val_log_spec_feature = np.asarray(val_log_spec_feature)
    np.save('../feature/log_mel_spec/val.npy', val_log_spec_feature)

    pass
