import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# TODO: Add data normalization fucntionality to the code
class TextDictionary:
    def __init__(self, vocab_file_path):

        self.load_vocab_file(vocab_file_path)
        self.vocab_size = len(self.dict)

        self.token2int = {j: i for i, j in enumerate(self.dict)}
        self.int2token = {i: j for i, j in enumerate(self.dict)}

    def load_vocab_file(self, vocab_file_path):
        with open(vocab_file_path, "r") as f:
            self.dict = f.read().splitlines()

    def eos(self):
        return self.token2int['<EOS>']

    def sos(self):
        return self.token2int['<SOS>']

    def pad(self):
        return self.token2int['<PAD>']

    def get_int(self, token):
        return self.token2int[token]

    def get_token(self, index):
        return self.int2token[index]

    def __len__(self):
        return len(self.dict)


class FixedDataset(Dataset):
    def __init__(self, dataPaths, xtype=torch.float32, ytype=torch.long, x_transforms=None, y_transforms=None, sort=False):

        self.trainData = np.load(dataPaths["data"], allow_pickle=True)
        self.trainLabels = np.load(dataPaths["labels"], allow_pickle=True)

        self.dataSize = len(self.trainData)

        self.trainData = torch.tensor(self.trainData, dtype=xtype)
        if x_transforms:
            self.trainData = x_transforms(self.trainData)

        if self.trainLabels:
            self.trainLabels = torch.tensor(self.trainLabels, dtype=ytype)
            if y_transforms:
                self.trainLabels = y_transforms(self.trainLabels)
        else:
            self.trainLabels = torch.zeros(len(self.trainData), dtype=ytype)


    def __getitem__(self, index):
        return self.trainData[index], self.trainLabels[index]

    def __len__(self):
        return self.dataSize


class VariableDataset(Dataset):
    def __init__(self, dataPaths, xtype=torch.float32, ytype=torch.long, x_transforms=None, y_transforms=None, sort=False):

        self.trainData = np.load(dataPaths["data"], allow_pickle=True)
        self.trainLabels = np.load(dataPaths["labels"], allow_pickle=True) if dataPaths["labels"] else None

        self.trainData = self.trainData
        self.trainLabels = self.trainLabels

        self.dataSize = len(self.trainLabels)

        self.trainData = [torch.tensor(x, dtype=xtype) for x in self.trainData]
        if x_transforms:
            for transform in x_transforms:
                self.trainData = [transform(x) for x in self.trainData]

        if self.trainLabels is not None:
            self.trainLabels = [torch.tensor(y, dtype=ytype) for y in self.trainLabels]
            if y_transforms:
                for transform in y_transforms:
                    self.trainLabels = [transform(y) for y in self.trainLabels]
        else:
            self.trainLabels = torch.zeros(len(self.trainData), dtype=ytype)


    def __getitem__(self, index):
        return self.trainData[index], self.trainLabels[index]

    def __len__(self):
        return self.dataSize


def pad_collate(batch, batch_first=False, padding_value=0):

    data, target = zip(*batch)
    data_len = torch.tensor([len(x) for x in data], dtype=torch.long)
    target_len = torch.tensor([len(y) for y in target], dtype=torch.long)

    padded_data = pad_sequence(data, batch_first, padding_value)
    padded_target = pad_sequence(target, batch_first, padding_value)

    return padded_data, padded_target, data_len, target_len