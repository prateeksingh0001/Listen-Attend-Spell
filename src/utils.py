import os
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


def setupExperimentDirectory(experimentName):
    if not os.path.exists(experimentName):
        os.mkdir(experimentName)

    if not os.path.exists(os.path.join(experimentName, "logs/")):
        os.mkdir(os.path.join(experimentName, "logs/"))



def collatePad(batch):
    data, target = zip(*batch)

    data_len = torch.LongTensor([len(x) for x in data])
    target_len = torch.LongTensor([len(y) for y in target])

    padded_data = pad_sequence(data)
    padded_target = pad_sequence(target, batch_first=True)

    return padded_data, padded_target, data_len, target_len
