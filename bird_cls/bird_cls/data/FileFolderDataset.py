import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np

class FileFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = []
        self.labels = []

        # 遍历文件夹，获取文件路径和对应标签
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                for filename in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, filename)
                    self.file_paths.append(file_path)
                    self.labels.append(int(subdir))  # 文件夹名作为标签

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path = self.file_paths[idx]
        cwt_spec = np.load(file_path)

        if self.transform:
            transformer_cwt_spec = self.transform(cwt_spec)

        label = self.labels[idx]
        return transformer_cwt_spec, label


if __name__ == '__main__':
    # 示例用法
    root_directory = '/home/dv/Downloads/436773567/New_CWT/train'
    transform = transforms.Compose([transforms.ToTensor(),])
    dataset = FileFolderDataset(root_dir=root_directory, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    # 使用 DataLoader 迭代数据集
    for cwt_spec, labels in dataloader:
        # 在此处添加你的训练或测试代码，比如输入模型等
        pass
