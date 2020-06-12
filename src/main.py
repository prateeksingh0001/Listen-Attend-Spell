import os
import yaml
from yaml import Loader
import argparse

from data_loader.dataset import TextDictionary, VariableDataset, pad_collate
from model.las import LAS
from utils import collatePad, setupExperimentDirectory

from trainer.train import Seq2SeqTrainer
import torch
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="config file path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config_file, "r") as file:
        params = yaml.load(file, Loader=Loader)

    experimentPath = os.path.join(params["experimentFolder"], params["experimentName"])
    setupExperimentDirectory(experimentPath)

    writer = SummaryWriter(os.path.join(experimentPath, "logs/"))

    # define train and validation loader
    trainDataset = VariableDataset(params["data"]["train"])
    trainDataLoader = data.DataLoader(trainDataset,
                                      collate_fn=pad_collate,
                                      **params["data_loader"]["train"])

    valDataset = VariableDataset(params["data"]["val"])
    valDataLoader = data.DataLoader(valDataset,
                                    collate_fn=pad_collate,
                                    **params["data_loader"]["val"])

    target_dictionary = TextDictionary(params["data"]["vocab_file_path"])

    print("vocab_size: ", len(target_dictionary))
    print("dict = ", target_dictionary.dict)
    print("transform = ", target_dictionary.token2int)

    model = eval(params["arch"]["name"])(params["arch"]["args"], target_dictionary)
    model.to(device)
    print(model)

    criterion = eval(params["criterion"]["name"])(ignore_index=target_dictionary.pad())
    optimizer = eval(params["optimizer"]["name"])(model.parameters(), **params["optimizer"]["args"])
    lr_scheduler = eval(params["lr_scheduler"]["name"])(optimizer, **params["lr_scheduler"]["args"])

    trainer = eval(params["trainer"]["name"])(model, criterion, optimizer, lr_scheduler)
    trainer.train(valDataLoader, valDataLoader, target_dictionary, **params["trainer"]["train_params"])
