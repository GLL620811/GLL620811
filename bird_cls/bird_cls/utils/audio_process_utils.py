import torch
from torchaudio import transforms
import pywt
import numpy as np
from matplotlib import pyplot as plt
import librosa
from PIL import Image
import io


def process_audio_batch(batch_waveforms, nfft):
    transform = transforms.MelSpectrogram(sample_rate=16000, n_fft=nfft, hop_length=int(nfft * 0.25), win_length=nfft,
                                         n_mels=40)
    B, T = batch_waveforms.size()
    # STFT
    spectrograms = torch.stft(batch_waveforms,
                              n_fft=nfft,
                              win_length=nfft,
                              hop_length=int(nfft * 0.25),
                              return_complex=True)
    # # 获取梅尔频谱图的最小值和最大值
    epsilon = 1e-6
    # # 将梅尔频谱图归一化到 [0, 1] 范围
    log_power_spectrogram = 10 * torch.log10(torch.abs(spectrograms) + epsilon).unsqueeze(dim=1)
    normalized_features = (log_power_spectrogram - log_power_spectrogram.mean() + 1e-12) / (
             log_power_spectrogram.std() + 1e-12)
    # mel_spectrogram = transform(batch_waveforms)
    # log_mel_spec = torch.log10(mel_spectrogram + epsilon).unsqueeze(dim=1)
    # normalized_log_mel_spec_features = (log_mel_spec - log_mel_spec.mean() + 1e-12) / (
    #          log_mel_spec.std() + 1e-12)
    return normalized_features


# def TimeFrequencyCWT(data, fs, totalscal, wavelet='cmor3-3'):
#     # 采样数据的时间维度
#     t = np.arange(data.shape[0]) / fs
#     # 中心频率
#     wcf = pywt.central_frequency(wavelet=wavelet)
#     # 计算对应频率的小波尺度
#     cparam = 2 * wcf * totalscal
#     scales = cparam / np.arange(totalscal, 1, -1)
#     # 连续小波变换
#     [cwtmatr, frequencies] = pywt.cwt(data, scales, wavelet, 1.0 / fs)
#     # 绘图
#     plt.figure(figsize=(8, 4))
#     plt.contourf(t, frequencies, abs(cwtmatr), cmap='jet')
#     plt.ylabel(u"freq(Hz)")
#     plt.xlabel(u"time(s)")
#     plt.subplots_adjust(hspace=0.4)
#     plt.show()
#
#     # 绘制频谱图
#     plt.figure(figsize=(8, 4))
#     plt.imshow(abs(cwtmatr), aspect='auto', cmap='jet',
#                extent=[0, 1, frequencies.min(), frequencies.max()])
#     plt.ylabel("Frequency (Hz)")
#     plt.xlabel("Time (s)")
#     plt.colorbar(label='Magnitude')
#     plt.title('Spectrogram')
#
#     plt.subplots_adjust(hspace=0.4)
#     plt.show()
def TimeFrequencyCWT(data, fs=16000, totalscal=1000, wavelet='cmor3-3', epsilon=1e-6):
    # 中心频率
    wcf = pywt.central_frequency(wavelet=wavelet)
    # 计算对应频率的小波尺度
    cparam = 2 * wcf * totalscal
    scales = cparam / np.arange(totalscal, 1, -1)
    # 连续小波变换
    [cwtmatr, frequencies] = pywt.cwt(data, scales, wavelet, 1.0 / fs)
    log_cwtmatr_spec = np.log10(abs(cwtmatr) + epsilon)
    # plt.figure(figsize=(8, 4))
    # plt.imshow(log_cwtmatr_spec, aspect='auto', cmap='jet',
    #                extent=[0, 1, frequencies.min(), frequencies.max()])
    # plt.ylabel("Frequency (Hz)")
    # plt.xlabel("Time (s)")
    # plt.colorbar(label='Magnitude')
    # plt.title('Spectrogram')
    #
    # plt.subplots_adjust(hspace=0.4)
    # plt.show()
    fig, ax = plt.subplots()
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.imshow(log_cwtmatr_spec, aspect='auto', cmap='jet')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpg')
    plt.close(fig)
    image = Image.open(buffer)
    return image


def NewTimeFrequencyCWT(data, fs=16000, totalscal=258, wavelet='cmor3-3',nfft=512, epsilon=1e-6):
    frames = librosa.util.frame(data, frame_length=257, hop_length=int(nfft * 0.25),axis=0).transpose(1, 0)
    # 中心频率
    wcf = pywt.central_frequency(wavelet=wavelet)
    cparam = 2 * wcf * totalscal
    scales = cparam / np.arange(totalscal, 1, -1)
    [cwtmatr, frequencies] = pywt.cwt(frames, scales, wavelet, 1.0 / fs)
    log_cwtmatr_spec = np.log10(abs(cwtmatr).mean(-1) + epsilon)
    return log_cwtmatr_spec


def process_audio_batch_Morlet(batch_waveforms, transform, wavelet='cmor3-3', level=1000):
    log_cwtmatr_specs = []
    for i in range(batch_waveforms.shape[0]):
        log_cwtmatr_spec_image = TimeFrequencyCWT(batch_waveforms[i, ...].cpu().detach().numpy(), fs=16000,
                                                  totalscal=level,
                                                  wavelet=wavelet)
        transform_spec = transform(log_cwtmatr_spec_image)
    return torch.vstack(log_cwtmatr_specs)


def add_gaussian_noise_torch(signals, snr_db):
    """
    向PyTorch tensor中的信号批次添加高斯噪声。

    参数:
    signals -- 原始信号的批次，PyTorch tensor，形状为[batch_size, signal_length]。
    snr_db -- 信噪比，以分贝为单位。

    返回:
    带有高斯噪声的信号批次，PyTorch tensor。
    """
    # 计算信号功率和噪声功率
    signal_power = torch.mean(signals ** 2, dim=1, keepdim=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    # 生成高斯噪声
    mean_noise = 0
    # 注意这里的修改：我们现在使用std的平方根，并直接生成和signals同形状的噪声
    std_noise = noise_power.sqrt()
    noise = torch.normal(mean=mean_noise, std=std_noise.repeat(1, signals.shape[1]))

    # 向信号添加噪声
    noisy_signals = signals + noise
    return noisy_signals


if __name__ == '__main__':
    wave, sr = librosa.load('/home/dv/Downloads/436773567/BirdsData-16k/0009/373787_3.wav', sr=None)
    TimeFrequencyCWT(wave, fs=16000, totalscal=1000, wavelet='cmor3-3')
