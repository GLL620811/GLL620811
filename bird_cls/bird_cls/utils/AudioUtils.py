import torch
from torchaudio import transforms


class AudioUtils:
    def __init__(self):
        self.transform = None

    def process_audio_batch_ToLogSpec(self, batch_waveforms, nfft, epsilon=1e-8):
        B, T = batch_waveforms.size()
        spectrograms = torch.stft(batch_waveforms,
                                  n_fft=nfft,
                                  win_length=nfft,
                                  hop_length=int(nfft * 0.25),
                                  return_complex=True)

        # # 将梅尔频谱图归一化到 [0, 1] 范围
        log_power_spectrogram = 10 * torch.log10(torch.abs(spectrograms) + epsilon).unsqueeze(dim=1)
        return log_power_spectrogram.float()

    def process_audio_batch_ToLogMelSpec(self, batch_waveforms, nfft, n_mels, epsilon=1e-8):
        if self.transform is None:
            self.transform = transforms.MelSpectrogram(sample_rate=16000, n_fft=nfft, hop_length=int(nfft * 0.25),
                                                       win_length=nfft,
                                                       n_mels=n_mels)
        mel_spectrogram = self.transform(batch_waveforms)
        log_mel_spec = torch.log10(mel_spectrogram + epsilon).unsqueeze(dim=1)
        return (log_mel_spec.float() - log_mel_spec.min()) / (log_mel_spec.max() -  log_mel_spec.min())
