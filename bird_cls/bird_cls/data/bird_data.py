import numpy as np
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import json
import math
import torch


class BirdAudioDataSet(Dataset):
    """
    json_dir:数据集路径
    sample_rate:采样频率
    segment:一段音频切成多少段
    """

    def __init__(self, json_dir, sample_rate, segment):
        with open(json_dir, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        self.json_data = json_data
        self.sample_rate = sample_rate
        self.segment = segment

    def process_audio_file(self, index):
        file_path = self.json_data[index][0]
        cls = int(self.json_data[index][1])
        lens = int(self.json_data[index][2])
        sample_rate = self.json_data[index][3]
        wave_start = int(self.json_data[index][4])
        wave_end = int(self.json_data[index][5])
        # noise_wav_file = self.json_data[index][4]

        with sf.SoundFile(file_path) as audio_file:
            audio_file.seek(wave_start)
            target_lens = wave_end - wave_start
            assert target_lens == math.floor(self.sample_rate * self.segment),'Target lens is Error'
            waveform = audio_file.read(target_lens)
        # with sf.SoundFile(noise_wav_file) as audio_file:
        #     audio_file.seek(0)
        #     noise_wav_from =  audio_file.read(target_lens)
        # mixture_wavform = waveform + noise_wav_from
        # return waveform.T,mixture_wavform.T, target_lens,cls
        return waveform.T, target_lens, cls

    def __getitem__(self, index):
        waveform, target_lens, cls = self.process_audio_file(index)
        # return torch.from_numpy(waveform).float(), torch.from_numpy(mixture_wavform).float(), torch.tensor(target_lens).long(),torch.tensor(cls).long()
        return torch.from_numpy(waveform).float(), torch.tensor(target_lens).long(), torch.tensor(cls).long()

    def __len__(self):
        return len(self.json_data)


if __name__ == '__main__':
    batch_size = 32
    train_bird_dataset = BirdAudioDataSet(
        json_dir='/home/dw/.jupyter_data/downloads/436773567/BirdsData-16k/train_datas.json', sample_rate=16000,
        segment=2)
    train_loader = DataLoader(dataset=train_bird_dataset, batch_size=batch_size, shuffle=True, num_workers=0 / 1)
    eval_bird_dataset = BirdAudioDataSet(
        json_dir='/home/dw/.jupyter_data/downloads/436773567/BirdsData-16k/eval_datas.json', sample_rate=16000,
        segment=2)
    eval_loader = DataLoader(dataset=eval_bird_dataset, batch_size=batch_size, shuffle=False, num_workers=0 / 1)
