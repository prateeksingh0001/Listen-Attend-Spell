import os
import numpy as np
from tqdm import tqdm
import torch
import time
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from Levenshtein import distance as levenshtein_distance
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


# TODO: Move criterion from init to train function,
#  then we might then be able to find losses with different
#  loss functions without needing to make different training classes.

# TODO: Add visualization functionality to the classes

# TODO: Use eval metric as function parameter and should not be
#  like right now with levenshtein distance

# TODO: Add attention visualization to it.
class BaseTrainer:
    def __init__(self, model, criterion, optimizer, scheduler, checkpoint=None, device=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = 0

        self.device = device if device else self.get_device()

        if checkpoint:
            self.load_checkpoint(checkpoint)

    def load_checkpoint(self, checkpoint_path):

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

            if self.optimizer:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # TODO: While saving the model save the scheduler state dict also
            # if self.scheduler:
            #     self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            self.epoch = checkpoint["epoch"]

        except FileNotFoundError:
            print("Provided checkpoint path %s not found" % checkpoint_path)
            exit()

    def get_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Torch to device set to %s" % device)

        return device

    def train(self, train_data_loader, dev_data_loader=None, target_dict=None, num_epochs=50, save_dir="./",
              save_frequency=1, report_frequency=1, inference_params=None):

        assert inference_params is not None, "provide parameters for inference"
        assert self.optimizer is not None, "no optimizer present"
        assert self.criterion is not None, "no criterion present"
        assert target_dict is not None, "target dict is none"

        self.writer = SummaryWriter(os.path.join(save_dir, "logs/"))

        self.model.to(self.device)
        self.criterion.to(self.device)

        print("Training set batches %d, Batch size: %d" % (len(train_data_loader), train_data_loader.batch_size))

        if dev_data_loader:
            print("Dev set batches %d, Batch size: %d" % (len(dev_data_loader), dev_data_loader.batch_size))
        else:
            print("No dev loader present")

        print("Training starting at: %s" % datetime.now())

        for epoch_num in range(self.epoch + 1, self.epoch + num_epochs + 1):

            print("=" * 50, "Epoch: %d" % epoch_num, "=" * 50)

            epoch_train_loss = self.train_epoch(train_data_loader, epoch_num)

            if dev_data_loader:
                epoch_dev_loss, epoch_dev_metric = self.eval_epoch(dev_data_loader, target_dict, inference_params)
            if self.scheduler:
                self.scheduler.step(epoch_dev_loss)

            if save_frequency and epoch_num % save_frequency == 0:
                self.save_checkpoint(epoch_num, epoch_dev_metric, save_dir)

            if report_frequency and epoch_num % report_frequency == 0:
                self.log_epoch(epoch_num, epoch_train_loss, epoch_dev_loss, epoch_dev_metric)

    def train_epoch(self, epoch_num, train_data_loader):

        self.model.train()
        progress_bar = tqdm(total=len(train_data_loader), position=0, leave=True)

        running_loss = 0
        for batch_idx, batch in enumerate(train_data_loader):
            self.optimizer.zero_grad()

            loss = self.compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)
            self.optimizer.step()
            running_loss += loss.item()

            progress_bar.update(1)

            del batch
            torch.cuda.empty_cache()

        progress_bar.close()

        return running_loss / len(train_data_loader)

    def eval_epoch(self, dev_data_loader, target_dict, inference_parameters):
        self.model.eval()

        dev_dataset_len = len(dev_data_loader)
        running_dev_loss = 0
        running_dev_metric = 0

        progress_bar = tqdm(total=len(dev_data_loader), position=0, leave=True)

        with torch.no_grad():
            for batch_idx, batch in enumerate(dev_data_loader):
                running_dev_loss += self.compute_loss(batch)
                running_dev_metric += self.compute_metric(batch, target_dict, inference_parameters)

                progress_bar.update(1)

                del batch
                torch.cuda.empty_cache()

        progress_bar.close()
        return running_dev_loss / dev_dataset_len, running_dev_metric / dev_dataset_len

    def compute_loss(self, batch, criterion, **kwargs):
        raise NotImplementedError

    def compute_metric(self, batch, eval_metric, **kwargs):
        raise NotImplementedError

    def get_outputs(self, batch):
        raise NotImplementedError

    def get_predictions(self, batch, **kwargs):
        raise NotImplementedError

    def save_checkpoint(self, epoch, epoch_dev_accuracy, save_dir):

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": None,
            "epoch_dev_accuracy": epoch_dev_accuracy
        }

        torch.save(checkpoint, os.path.join(save_dir, "epoch_%d_dev_accuracy_%d"%(epoch, epoch_dev_accuracy)))

    def log_epoch(self, epoch, epoch_train_loss, epoch_dev_loss,
                  epoch_dev_metric):
        epoch_log = "Epoch: {} \t train_loss: {} \t val_loss: {} \t val_metric: {}".format(
            epoch, epoch_train_loss, epoch_dev_loss, epoch_dev_metric)

        print(epoch_log)

        self.writer.add_scalar("Loss/train", epoch_train_loss, global_step=epoch)
        self.writer.add_scalar("Loss/eval", epoch_dev_loss, global_step=epoch)
        self.writer.add_scalar("Lev_dis/eval", epoch_dev_metric, global_step=epoch)


class Seq2SeqTrainer(BaseTrainer):
    def __init__(self, model, criterion, optimizer, scheduler, checkpoint=None, device=None):
        super(Seq2SeqTrainer, self).__init__(model, criterion, optimizer, scheduler, checkpoint, device)

    def train_epoch(self, train_data_loader, epoch_num):
        sample_prob = min((epoch_num / 100) * 2, 0.5)  # Probability for teacher forcing

        self.model.train()
        progress_bar = tqdm(total=len(train_data_loader), position=0, leave=True)

        running_loss = 0
        for batch_idx, batch in enumerate(train_data_loader):
            self.optimizer.zero_grad()

            loss = self.compute_loss(batch, sample_prob=sample_prob)
            running_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)
            self.optimizer.step()

            del batch
            torch.cuda.empty_cache()

            progress_bar.update(1)
        progress_bar.close()

        for batch in train_data_loader:
            self.visualize_attention(batch, sample_prob=1.0, num_visualizations=1)
            break

        return running_loss / len(train_data_loader)

    def compute_loss(self, batch, **kwargs):
        """
        :param batch: Contains data, target, data_len, target_len with the shapes as
                      data
        :param kwargs: Contains the sample_prob argument, the sample_prob would be used
                       during training but would not be used during inference.
        :return: Average loss per token for each batch
        """
        data, target, data_len, target_len = batch
        data, target, data_len, target_len = data.to(self.device), target.to(self.device), data_len.to(self.device), \
                                             target_len.to(self.device)

        # TODO: Find out why target is being truncated
        logits, srclens, attns = self.model(data, data_len, target[:-1], kwargs.get("sample_prob", 0.00))

        vocab_size = logits.shape[2]

        # Targets need to be shifted by 1 as the targets start with <SOS>
        # Eg. If the target is <SOS> T H E  A R E ... <EOS>, then for calculating the loss
        # we will consider the targets from T and not <SOS>, we must make the appropriate
        # changes to the target lengths.
        # Since the criterion has the reduction function as mean it will ignore all the
        # padding index character and calculate the mean over all the tokens.
        avg_loss = self.criterion(logits.view(-1, vocab_size), target[1:].view(-1))
        target_len -= 1

        del data, target, data_len, target_len
        torch.cuda.empty_cache()

        return avg_loss

    def compute_metric(self, batch, target_dict, inference_params):

        data, target, data_len, target_len = batch
        data, data_len, target, target_len = data.to(self.device), data_len.to(self.device), target.to(self.device), \
                                             target_len.to(self.device)

        padded_seq, seq_len, srclen, attns = self.model.predict(data, data_len, **inference_params)

        print("Padded seq shape: ", padded_seq.shape)

        metric = self.calculate_levenshtein_dist(padded_seq, seq_len, target, target_len, levenshtein_distance,
                                                 target_dict)

        del data, data_len
        return metric

    def calculate_levenshtein_dist(self, padded_seq, seq_len, target, target_len, eval_metric, target_dict):
        batch_size = len(target_len)
        distances = []
        pred_seq_string = self.index2string(padded_seq, seq_len, target_dict)
        target_seq_string = self.index2string(target, target_len, target_dict)

        for i in range(batch_size):
            if i == 0:
                print("prediction: ", pred_seq_string[i])
                print("target: ", target_seq_string[i])

            distance = eval_metric(pred_seq_string[i], target_seq_string[i])
            distances.append(distance)

        print(distances)
        return np.mean(distances)

    def index2string(self, padded_seq, seq_len, target_dict, batch_first=False):
        batch_size = len(seq_len)
        seq_strings = []

        for i in range(batch_size):
            seq_tokens = padded_seq[i, :seq_len[i]] if batch_first else padded_seq[:seq_len[i], i]
            seq_strings.append(''.join([target_dict.int2token[x.item()] for x in seq_tokens[1:-1]]))

        return seq_strings

    def visualize_attention(self, batch, sample_prob=1.0, num_visualizations=1):

        data, target, data_len, target_len = batch
        data, target, data_len, target_len = data.to(self.device), target.to(self.device), data_len.to(self.device), \
                                             target_len.to(self.device)

        # TODO: Find out why target is being truncated
        logits, srclens, attns = self.model(data, data_len, target[:-1], sample_prob)

        target = target[1:, :]
        target_len = target_len - 1


        target_len = target_len.cpu().detach().numpy()  # (N, )
        srclens = srclens.cpu().detach().numpy()  # (N, )
        attns = attns.cpu().detach().numpy()  # (T, N, S)
        T, N, S = attns.shape
        num_visualizations = min(num_visualizations, N) if num_visualizations > -1 else N
        for i in range(num_visualizations):
            fig, ax = plt.subplots(figsize=(srclens[i] * 0.1, target_len[i] * 0.1))
            ax.imshow(attns[:target_len[i], i, :srclens[i]])
            ax.set_xticks(np.arange(0, srclens[i], 4))
            ax.set_yticks(np.arange(0, target_len[i], 2))
            ax.set_xlabel('Input time steps')
            ax.set_ylabel('Output time steps')
            # ax.set_xticklabels(list(data[i][0]))
            # ax.set_yticklabels(data[i][1].split() + ['</s>'])
            # ax.set_ylim(target_lens[i] - 0.5, -0.5)
            plt.show()