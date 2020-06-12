import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

class TorchGadget():
    def __init__(self, model, optimizer=None, scheduler=None, checkpoint='', device=None):
        """Instantiate TorchGadget with torch objects and optionally a checkpoint filename"""
        self.device = device if device else self.get_device()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = 0
        self.train_loss = None
        self.train_metric = None
        self.dev_loss = None
        self.dev_metric = None

        if checkpoint:
            self.load_checkpoint(checkpoint)


    def __repr__(self):
        str_joins = []
        str_joins.append(f"Device: {self.device}")
        str_joins.append(str(self.model))
        str_joins.append(str(self.optimizer) if self.optimizer else "No optimizer provided.")
        str_joins.append(str(self.scheduler) if self.scheduler else "No scheduler provided.")
        str_joins.append(f"Epoch: {self.epoch}")
        if self.train_loss:
            str_joins.append(f"Train loss: {self.train_loss}")
        if self.train_metric:
            str_joins.append(f"Train metric: {self.train_metric}")
        if self.dev_loss:
            str_joins.append(f"Development loss: {self.dev_loss}")
        if self.dev_metric:
            str_joins.append(f"Development metric: {self.dev_metric}")
        return '\n'.join(str_joins)


    def get_outputs(self, batch):
        """
        CUSTOMIZABLE FUNCTION - used as a helper function for compute_loss
        Unpacks a batch of data and runs it through the model's forward method to get outputs.
        Overload this function appropriately with your Dataset class's output and model forward function signature.
        """
        raise NotImplementedError


    def get_predictions(self, batch, **kwargs):
        """
        CUSTOMIZABLE FUNCTION - used as a helper function for compute_metric and predict_set
        Unpacks a batch of data and generates predictions from the model e.g. class labels for classification models.
        Overload this function appropriately with your Dataset class's output and model generation function signature.
        """
        raise NotImplementedError


    def compute_loss(self, batch, criterion):
        """
        CUSTOMIZABLE FUNCTION
        Computes the average loss over a batch of data for a given criterion (loss function).
        Overload this function appropriately with your Dataset class's output and model forward function signature.
        """
        raise NotImplementedError


    def compute_metric(self, batch, **kwargs):
        """
        CUSTOMIZABLE FUNCTION
        Computes the average evaluation metric over a batch of data.
        Overload this function appropriately with your Dataset class's output and model forward or inference function
        signature.
        """
        raise NotImplementedError


    def train(self, criterion, train_loader, dev_loader=None, eval_train=False, n_epochs=1000,
              save_dir='./', save_freq=1, report_freq=0, **kwargs):
        """
        Boiler plate training procedure, optionally saves the model checkpoint after every epoch
        :param train_loader: training set dataloader
        :param dev_loader: development set dataloader
        :param n_epochs: the epoch number after which to terminate training
        :param criterion: the loss criterion
        :param save_dir: the save directory
        :param report_freq: report training approx report_freq times per epoch
        """
        # Check config
        # torch.autograd.set_detect_anomaly(True)  # TODO DELETE
        assert self.optimizer is not None, "Optimizer required for training. Set TorchGadget.optimizer"
        if self.scheduler and not eval_train and not dev_loader:
            print("Warning: Use of scheduler without evaluating either the training or validation sets per epoch")
            print("If scheduler is dynamic, it only compares training loss over one reporting cycle (may be unstable).")

        save_dir = self.check_save_dir(save_dir)
        if save_dir[-1] != '/':
            save_dir = save_dir + '/'

        self.model.to(self.device)
        criterion.to(self.device)


        # Prepare to begin training
        batch_group_size = int(len(train_loader) / report_freq) if report_freq else 0  # report every batch_group_size

        print(f"Training set batches: {len(train_loader)}\tBatch size: {train_loader.batch_size}.")
        print(f"Development set batches: {len(dev_loader)}\tBatch size: {dev_loader.batch_size}." if dev_loader \
              else "No development set provided")

        print(f"Beginning training at {datetime.now()}")
        if self.epoch == 0:
            with open(save_dir + "results.txt", mode='a') as f:
                header = "epoch"
                if eval_train:
                    header = header + ",train_loss,train_metric"
                if dev_loader:
                    header = header + ",dev_loss,dev_metric"
                f.write(header + "\n")

        if self.epoch == 0 or (eval_train and (self.train_loss is None or self.train_metric is None)):
            self.train_loss = []
            self.train_metric = []
        if self.epoch == 0 or (dev_loader and (self.dev_loss is None or self.dev_metric is None)):
            self.dev_loss = []
            self.dev_metric = []


        # Train
        self.model.train()
        for epoch in range(self.epoch + 1, n_epochs + 1):
            # Train over epoch
            last_batch_group_loss = self.train_epoch(train_loader, criterion, epoch, batch_group_size)

            # Evaluate epoch
            with open(save_dir + "results.txt", mode='a') as f:
                line = str(epoch)
                if eval_train:  # evaluate over training set
                    epoch_train_loss = self.eval_set(train_loader, self.compute_loss, criterion=criterion)
                    epoch_train_metric = self.eval_set(train_loader, self.compute_metric, **kwargs)
                    self.train_loss.append(epoch_train_loss)
                    self.train_metric.append(epoch_train_metric)
                    line = line + f",{epoch_train_loss},{epoch_train_metric}"
                if dev_loader:  # evaluate over development set
                    epoch_dev_loss = self.eval_set(dev_loader, self.compute_loss, criterion=criterion)
                    epoch_dev_metric = self.eval_set(dev_loader, self.compute_metric, **kwargs)
                    self.dev_loss.append(epoch_dev_loss)
                    self.dev_metric.append(epoch_dev_metric)
                    line = line + f",{epoch_dev_loss},{epoch_dev_metric}"
                else:
                    epoch_dev_metric = None
                f.write(line + "\n")

            # Step scheduler
            if self.scheduler:
                if dev_loader:
                    self.try_sched_step(epoch_dev_loss)
                elif eval_train:
                    self.try_sched_step(epoch_train_loss)
                else:
                    self.try_sched_step(last_batch_group_loss)

            # Save checkpoint
            if save_freq and epoch % save_freq == 0:
                self.save_checkpoint(epoch, epoch_dev_metric, save_dir)

            # Print epoch log
            if dev_loader:
                self.print_epoch_log(epoch, epoch_dev_loss, epoch_dev_metric, dataset='dev')
            elif eval_train:
                self.print_epoch_log(epoch, epoch_train_loss, epoch_train_metric, dataset='train')
            else:
                self.print_epoch_log(epoch)

        print(f"Finished training at {datetime.now()}")


    def train_epoch(self, train_loader, criterion, epoch_num=1, batch_group_size=0):
        """
        TODO
        """
        # torch.autograd.set_detect_anomaly(True)

        avg_loss = 0.0  # Accumulate loss over subsets of batches for reporting
        for i, batch in enumerate(train_loader):
            batch_num = i + 1
            self.optimizer.zero_grad()

            loss = self.compute_loss(batch, criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()

            # Accumulate loss for reporting
            avg_loss += loss.item()
            if batch_group_size and (batch_num) % batch_group_size == 0:
                print(f'Epoch {epoch_num}, Batch {batch_num}\tTrain loss: {avg_loss / batch_group_size:.4f}\t{datetime.now()}')
                avg_loss = 0.0

            # Cleanup
            del batch
            torch.cuda.empty_cache()

        return avg_loss


    def save_checkpoint(self, epoch_num, epoch_dev_metric, save_dir):
        """Saves a training checkpoint"""
        checkpoint = {
            'epoch': epoch_num,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': getattr(self.scheduler, 'state_dict', lambda: None)(),
            'train_loss': self.train_loss,
            'train_metric': self.train_metric,
            'dev_loss': self.dev_loss,
            'dev_metric': self.dev_metric
        }
        if epoch_dev_metric:
            torch.save(checkpoint, save_dir + f"checkpoint_{epoch_num}_{epoch_dev_metric:.4f}.pth")
        else:
            torch.save(checkpoint, save_dir + f"checkpoint_{epoch_num}.pth")


    def print_epoch_log(self, epoch_num, loss=None, metric=None, dataset='dev'):
        """Prints a log of a training epoch at its completion"""
        epoch_log = f'Epoch {epoch_num} complete.'
        if loss is not None and metric is not None:
            epoch_log += f'\t{dataset} loss: {loss:.4f}\t{dataset} metric: {metric:.4f}'
        print(f'{epoch_log}\t{datetime.now()}')


    def eval_set(self, data_loader, compute_fn=None, **kwargs):
        """
        BOILERPLATE FUNCTION
        Evaluates the average evaluation metric (or other metric such as loss) of the model on a given dataset
        :param data_loader: A dataloader for the data over which to evaluate
        :param compute_fn: set to the compute_metric method by default (can be set to compute_loss method or any other)
        :param kwargs: either The criterion for compute_loss or other kwargs for compute_metric
        :return: The average loss or metric per sentence
        """
        if compute_fn is None:
            compute_fn = self.compute_metric
        self.model.eval()
        accum = 0.0
        batch_count = 0.0

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                num_data_points = len(batch)
                if i == 0:
                    batch_size = num_data_points  # assumes batch first ordering

                # Accumulate
                batch_fraction = num_data_points / batch_size  # last batch may be smaller than batch_size
                batch_count += batch_fraction
                accum += batch_fraction * compute_fn(batch, **kwargs)

                # Clean up
                del batch
                torch.cuda.empty_cache()

        self.model.train()

        return accum / batch_count


    def predict_set(self, data_loader, **kwargs):
        """
        BOILERPLATE FUNCTION
        Generates the predictions of the model on a given dataset
        :param data_loader: A dataloader for the data over which to evaluate
        :param kwargs: kwargs for the get_predictions method, if any
        :return: Concatenated predictions of the same type as returned by self.get_predictions
        """
        self.model.eval()
        accum = []

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                predictions_batch, _ = self.get_predictions(batch, **kwargs)
                # predictions_batch = outputs.detach().to('cpu')
                accum.extend(predictions_batch)

                # Clean up
                del batch
                torch.cuda.empty_cache()

        self.model.train()

        if isinstance(predictions_batch, list):
            predictions_set = accum
        elif isinstance(predictions_batch, torch.Tensor):
            predictions_set = torch.cat(accum)
        elif isinstance(predictions_batch, np.ndarray):
            predictions_set = np.concatenate(accum)
        else:
            raise NotImplementedError("Unsupported output type: ", type(predictions_batch))

        del predictions_batch
        torch.cuda.empty_cache()

        return predictions_set


    def training_plot(self, plot_loss=True, plot_metric=True):
        if plot_loss and plot_metric:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=False)
        else:
            fig, ax1 = plt.subplots()
        plt.title('Training plot')
        plt.xlabel('Epoch')

        if plot_loss:
            if self.train_loss:
                ax1.plot(range(1, self.epoch + 1), self.train_loss, 'b-', label='Training loss')
            if self.dev_loss:
                ax1.plot(range(1, self.epoch + 1), self.dev_loss, 'g-', label='Development loss')
            ax1.set_ylabel('Loss')
            ax1.legend()
        if plot_metric:
            ax = ax2 if plot_loss else ax1
            if self.train_metric:
                ax.plot(range(1, self.epoch + 1), self.train_metric, 'b-', label='Training metric')
            if self.dev_metric:
                ax.plot(range(1, self.epoch + 1), self.dev_metric, 'g-', label='Development metric')
            ax.set_ylabel('Metric')
            ax.legend()
        plt.show()


    def load_checkpoint(self, checkpoint_path=''):
        """Loads a checkpoint for the model, epoch number and optimizer if provided"""
        # Try loading checkpoint and keep asking for valid checkpoint paths upon failure.
        done = False
        while not done:
            if checkpoint_path == 'init':
                self.epoch = 0
                done = True
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if self.optimizer:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except (AttributeError, KeyError):
                    pass
                self.epoch = checkpoint['epoch']
                if 'dev_loss' in checkpoint:
                    self.dev_loss = checkpoint['dev_loss']
                if 'dev_metric' in checkpoint:
                    self.dev_metric = checkpoint['dev_metric']
                done = True
            except FileNotFoundError:
                print(f"Provided checkpoint path {checkpoint_path} not found. Path must include the filename itself.")
                checkpoint_path = input("Provide a new path, or [init] to use randomized weights: ")


    def try_sched_step(self, metrics):
        """
        Steps the scheduler
        First tries to step with the metric in case the scheduler is dynamic, then falls back to a step without args.
        """
        try:
            self.scheduler.step(metrics=metrics)
        except TypeError:
            self.scheduler.step()


    def check_save_dir(self, save_dir):
        """Checks that the provided save directory exists and warns the user if it is not empty"""
        done = False
        while not done:
            if save_dir == 'exit':
                print("Exiting")
                sys.exit(0)

            #  Make sure save_dir exists
            if not os.path.exists(save_dir):
                print(f"Save directory {save_dir} does not exist.")
                mkdir_save = input(
                    "Do you wish to create the directory [m], enter a different directory [n], or exit [e]? ")
                if mkdir_save == 'm':
                    os.makedirs(save_dir)
                    done = True
                    continue
                elif mkdir_save == 'n':
                    save_dir = input("Enter new save directory: ")
                    continue
                elif mkdir_save == 'e':
                    sys.exit()
                else:
                    print("Please enter one of [m/n/e]")
                    continue

            #  Ensure user knows if save_dir is not empty
            if os.listdir(save_dir):
                use_save_dir = input(
                    f"Save directory {save_dir} is not empty. Do you want to overwrite it? [y/n] ")
                if use_save_dir == 'n':
                    save_dir = input("Enter new save directory (or [exit] to exit): ")
                elif use_save_dir == 'y':
                    sure = input(f"Are you sure you want to overwrite the directory {save_dir}? [y/n] ")
                    if sure == 'y':
                        import shutil
                        assert save_dir != '~/' and save_dir != '~'
                        shutil.rmtree(save_dir)
                        os.makedirs(save_dir)
                        done = True
                    elif sure != 'n':
                        print("Please enter one of [y/n]")
                else:
                    print("Please enter one of [y/n]")
            else:
                done = True

        return save_dir


    def get_device(self):
        """Gets the preferred torch.device and warns the user if it is cpu"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device == 'cpu':
            use_cpu = input("Torch device is set to 'cpu'. Are you sure you want to continue? [y/n] ")
            while use_cpu not in ('y', 'n'):
                print("Please input y or n")
                use_cpu = input("Torch device is set to 'cpu'. Are you sure you want to continue? [y/n] ")
            if use_cpu == 'n':
                sys.exit()
        return device


class ClassificationGadget(TorchGadget):
    def __init__(self, *args, **kwargs):
        super(ClassificationGadget, self).__init__(*args, **kwargs)

    def get_outputs(self, batch):
        """
        Takes a batch and runs it through the model's forward method to get outputs.
        Overload this function appropriately with your Dataset class's output and model forward function signature
        """
        # Unpack batch
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        # Compute outputs
        outputs = self.model(x)

        # Clean up
        del x
        # no need to del batch (deleted in the calling function)
        torch.cuda.empty_cache()

        return outputs, y

    def get_predictions(self, batch, **kwargs):
        """
        Takes a batch and generates predictions from the model e.g. class labels for classification models
        Overload this function appropriately with your Dataset class's output and model generation function signature
        """
        # Unpack batch
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        # Generate predictions
        outputs = self.model(x)
        _, pred_labels = outputs.max(dim=1)
        pred_labels = pred_labels.view(-1)

        # Clean up
        del x, outputs
        # no need to del batch (deleted in the calling function)
        torch.cuda.empty_cache()

        return pred_labels, y

    def compute_loss(self, batch, criterion):
        """
        Computes the average loss over a batch of data for a given criterion (loss function).
        Overload this function appropriately with your Dataset class's output and model forward function signature
        """
        outputs, target = self.get_outputs(batch)
        loss = criterion(outputs, target)

        # Clean up
        del outputs, target
        # no need to del batch (deleted in the calling function)
        torch.cuda.empty_cache()

        return loss

    def compute_metric(self, batch, **kwargs):
        """
        Computes the average evaluation metric over a batch of data.
        Overload this function appropriately with your Dataset class's output and model forward or inference function
        signature
        """
        predictions, target = self.get_predictions(batch)
        metric = self._accuracy(predictions, target)  # can be changed to any evaluation metric, with optional kwargs

        # Clean up
        del predictions, target
        # no need to del batch (deleted in the calling function)
        torch.cuda.empty_cache()

        return metric

    def _accuracy(self, pred_labels, labels):
        """Mean accuracy over predicted and true labels"""
        batch_size = len(labels)
        accuracy = torch.sum(torch.eq(pred_labels, labels)).item() / batch_size
        return accuracy


class Seq2SeqGadget(TorchGadget):
    def __init__(self, *args, **kwargs):
        super(Seq2SeqGadget, self).__init__(*args, **kwargs)
        from Levenshtein import distance as levenshtein_distance
        self.levenshtein_distance = levenshtein_distance

    def get_outputs(self, batch, sample_prob=0.0):
        """
        CUSTOMIZABLE FUNCTION
        Takes a batch and runs it through the model's forward method to get outputs.
        Overload this function appropriately with your Dataset class's output and model forward function signature
        """
        # Unpack batch
        x, y, xlens, ylens = batch
        x, y, xlens, ylens = x.to(self.device), y.to(self.device), xlens.to(self.device), ylens.to(self.device)

        # Compute outputs
        logits, srclens, attns = self.model(x, xlens, y[:-1, :], sample_prob)  # truncate final input time step

        # Clean up
        del x
        # no need to del batch (deleted in the calling function)
        torch.cuda.empty_cache()

        return logits, y[1:, :], ylens - 1, srclens, attns  # shift targets left


    def get_predictions(self, batch, max_len, beam_width=1):
        """
        CUSTOMIZABLE FUNCTION
        Takes a batch and generates predictions from the model e.g. class labels for classification models
        Overload this function appropriately with your Dataset class's output and model generation function signature
        """
        # Unpack batch
        x, y, xlens, ylens = batch
        x, y, xlens, ylens = x.to(self.device), y.to(self.device), xlens.to(self.device), ylens.to(self.device)

        # Generate predictions
        padded_seqs, seq_lens, srclens, attns = self.model.predict(x, xlens, max_len, beam_width)

        # Clean up
        del x, xlens
        # no need to del batch (deleted in the calling function)
        torch.cuda.empty_cache()

        return padded_seqs, y, seq_lens, ylens, srclens, attns

    def compute_loss(self, batch, criterion, sample_prob=0.0):
        """
        CUSTOMIZABLE FUNCTION
        Computes the average loss over a batch of data for a given criterion (loss function).
        Overload this function appropriately with your Dataset class's output and model forward function signature
        """
        logits, target, target_lens, srclens, attns = self.get_outputs(batch, sample_prob)

        vocab_size = logits.shape[2]
        total_loss = criterion(logits.view(-1, vocab_size), target.view(-1))  # flatten all time steps across batches
        total_tokens = target_lens.sum()
        loss = total_loss / total_tokens

        # Clean up
        del logits, target, target_lens, srclens, attns
        # no need to del batch (deleted in the calling function)
        torch.cuda.empty_cache()

        return loss

    def compute_metric(self, batch, max_len, index2token, beam_width=1):
        """
        CUSTOMIZABLE FUNCTION
        Computes the average evaluation metric over a batch of data for a given criterion (loss function).
        Overload this function appropriately with your Dataset class's output and model forward or inference function
        signature
        """
        padded_seqs, y, seq_lens, ylens, srclens, attns = self.get_predictions(batch, max_len, beam_width)
        metric = self.padded_levenshtein_distance(padded_seqs, y, seq_lens, ylens, index2token)

        # Clean up
        del padded_seqs, seq_lens, y, ylens, srclens, attns
        # no need to del batch (deleted in the calling function)
        torch.cuda.empty_cache()

        return metric


    def padded_levenshtein_distance(self, seqs, y, seq_lens, y_lens, index2token, batch_first=False):
        """
        :param seqs: (max_len, N) Batch input of generated text INCLUDING <sos> and <eos> tokens
        :param seq_lens: (N, ) Generated seq lengths INCLUDING <sos> and <eos> tokens
        :param y: (max_ylen, N) Batch input of target text INCLUDING <sos> and <eos> tokens
        :param y_lens: (N, ) Target text lengths INCLUDING <sos> and <eos> tokens
        :param index2token: the dictionary mapping indices to string tokens
        :param batch_first: Bool
        """
        N = len(y_lens)
        distances = []
        pred_seq_strings = self.idx2string(seqs, seq_lens, index2token, batch_first=batch_first)
        true_seq_strings = self.idx2string(y, y_lens, index2token, batch_first=batch_first)
        for i in range(N):
            if i == 0:  # print one prediction TODO DELETE
                print('hyp:', pred_seq_strings[i])
                print('ref:', true_seq_strings[i])

            distance = self.levenshtein_distance(pred_seq_strings[i], true_seq_strings[i])
            distances.append(distance)
        print(distances)
        return sum(distances) / N


    def idx2string(self, seqs, lens, index2token, batch_first=False):
        """
        Converts a (batch) of token indices in a tensor to a list of output strings
        :param seqs: (max_len, N) Batch input of target text INCLUDING <sos> and <eos> tokens
        :param lens: (N, ) generated seq lengths INCLUDING <sos> and <eos> tokens
        :param index2token: the dictionary mapping indices to string tokens
        :param batch_first: Bool
        """
        N = len(lens)
        seq_strings = []
        for i in range(N):
            # Strip <pad> tokens
            seq_tokens = seqs[i, :lens[i]] if batch_first else seqs[:lens[i], i]

            # Strip <sos> and <eos> tokens and convert tensors of indices into strings
            seq_str = ''.join([index2token[x.item()] for x in seq_tokens[1:-1]])
            seq_strings.append(seq_str)
        return seq_strings


    def visualize_attention(self, batch, sample_prob=1.0, num_visualizations=-1):
        """
        Visualizes the attention for a batch of data.
        :param sample_prob: scheduled sampling probability (set to 0.0 to teacher force; 1.0 for greedy search)
        """
        _, _, target_lens, srclens, attns = self.get_outputs(batch, sample_prob)
        target_lens = target_lens.cpu().detach().numpy()  # (N, )
        srclens = srclens.cpu().detach().numpy()  # (N, )
        attns = attns.cpu().detach().numpy()  # (T, N, S)
        T, N, S = attns.shape
        num_visualizations = min(num_visualizations, N) if num_visualizations > -1 else N
        for i in range(num_visualizations):
            fig, ax = plt.subplots(figsize=(srclens[i] * 0.1, target_lens[i] * 0.1))
            ax.imshow(attns[:target_lens[i], i, :srclens[i]])
            ax.set_xticks(np.arange(0, srclens[i], 4))
            ax.set_yticks(np.arange(0, target_lens[i], 2))
            ax.set_xlabel('Input time steps')
            ax.set_ylabel('Output time steps')
            # ax.set_xticklabels(list(data[i][0]))
            # ax.set_yticklabels(data[i][1].split() + ['</s>'])
            # ax.set_ylim(target_lens[i] - 0.5, -0.5)
            plt.show()

    def train_epoch(self, train_loader, criterion, epoch_num=1, batch_group_size=0):
        """
        TODO
        """
        sample_prob = min((epoch_num / 100) * 2, 0.5)
        print(f'Epoch {epoch_num}: using scheduled sampling probability of {sample_prob}.')

        avg_loss = 0.0  # Accumulate loss over subsets of batches for reporting
        for i, batch in enumerate(train_loader):
            batch_num = i + 1
            self.optimizer.zero_grad()

            loss = self.compute_loss(batch, criterion, sample_prob)  # EDITED HERE
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()

            # Accumulate loss for reporting
            avg_loss += loss.item()
            if batch_group_size and (batch_num) % batch_group_size == 0:
                print(f'Epoch {epoch_num}, Batch {batch_num}\tTrain loss: {avg_loss / batch_group_size:.4f}\t{datetime.now()}')
                avg_loss = 0.0

            # Cleanup
            del batch
            torch.cuda.empty_cache()

        for batch in train_loader:
            self.visualize_attention(batch, sample_prob=1.0, num_visualizations=1)
            break
        return avg_loss


    def print_epoch_log(self, epoch_num, loss=None, metric=None, dataset='dev'):
        epoch_log = f'Epoch {epoch_num} complete.'
        if loss is not None and metric is not None:
            epoch_log += f'\t{dataset} loss: {loss:.4f}\t{dataset} ppl: {np.exp(loss.cpu().detach().numpy()):.4f}\t{dataset} metric: {metric:.4f}'
        print(f'{epoch_log}\t{datetime.now()}')


    def predict_set(self, data_loader, max_len, index2token, beam_width=1):
        """
        Returns the a list of strings generated by get_predictions
        :param data_loader: A dataloader for the data over which to evaluate
        :param max_len: maximum output time steps
        :param index2token: the dictionary mapping indices to string tokens
        :return: Concatenated predictions of the same type as returned by self.get_predictions
        """
        self.model.eval()
        predictions_set = []

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                padded_seqs, _, seq_lens, _, _, _ = self.get_predictions(batch, max_len, beam_width)
                # predictions_batch = outputs.detach().to('cpu')
                predictions_set.extend(self.idx2string(padded_seqs, seq_lens, index2token))

                # Clean up
                del batch
                torch.cuda.empty_cache()

        self.model.train()

        del padded_seqs, seq_lens
        torch.cuda.empty_cache()

        return predictions_set


class BeamSearch():
    def __init__(self, max_len, pad_idx, sos_idx, eos_idx, batch_size, beam_width):
        """
        TODO
        """
        self.T = max_len
        self.N = batch_size
        self.K = beam_width
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.rnn_hidden_states = None

        # sum of log probability scores
        self.scores = None  # (N, K)
        # input token indices (i.e. <sos> is token 0)
        self.seqs = None  # (T, N, K)
        # sequence lengths
        self.lens = None  # (N, K)
        # mask of which beam candidates have not finished decoding
        self.active_mask = None  # (N, K)

    def step(self, logits, step_num=0):
        """
        Beam search step
        :param logits: (N * K, V)
        :param step_num: int
        """
        lprob = torch.nn.functional.log_softmax(logits, dim=-1)  # step == 0 ? (N, V) : (N * K, V)

        if step_num == 0:
            # Initialize beams
            self.N = len(logits)
            self.seqs = torch.full((self.T, self.N, self.K), fill_value=self.pad_idx, dtype=torch.long, device=logits.device)
            self.lens = torch.zeros((self.N, self.K), dtype=torch.long, device=logits.device)
            self.scores, self.seqs[step_num + 1] = lprob.topk(self.K, sorted=False)  # (N, K), (N, K)

            # Check if any candidates reached <eos> and update active_mask and lens
            self.active_mask = (self.seqs[step_num + 1] != self.eos_idx).clone().detach().bool()  # (N, K)
            self.lens[~self.active_mask] = step_num + 2
        else:
            # Add lprobs to scores
            lprob = lprob.reshape(self.N, self.K, -1)  # (N, K, V)
            return lprob
            all_cand_scores = self.scores.unsqueeze(-1) + lprob  # (N, K, V) broadcast scores over V

            # Loop over each of the N beams
            for i in range(self.N):
                k = self.active_mask[i].sum()  # active beam width
                if k > 0:
                    # Extend each seq in the beam by k candidates
                    cand_scores, tokens = all_cand_scores[i].topk(k, sorted=False)  # (K, k), (K, k)
                    cand_scores[~self.active_mask[i]] = float('-inf')  # ensure inactive candidates are not selected

                    # Prune down to top k beams for each instance in the batch
                    pruned_scores, flat_idxs = torch.flatten(cand_scores).topk(k, sorted=False)  # (k, ), (k, )
                    pruned_row_idxs = torch.floor(torch.div(flat_idxs, self.K).float()).long()  # the cands to extend
                    pruned_col_idxs = torch.fmod(flat_idxs, k)  # the tokens to extend them with
                    pruned_tokens = tokens[pruned_row_idxs, pruned_col_idxs]

                    # Update scores and seqs (note the use of nonzero().squeeze(1), which turns masks into indices)
                    active_idxs = self.active_mask[i].nonzero().squeeze(1)
                    self.scores[i, active_idxs] = pruned_scores
                    self.seqs[:step_num + 1, i, active_idxs] = self.seqs[:step_num + 1, i, pruned_row_idxs]  # base seqs
                    self.seqs[step_num + 1, i, active_idxs] = pruned_tokens  # extend the base seqs with the selected tokens

            # Check if any candidates reached <eos> and update active_mask and lens
            eos_mask = (self.seqs[step_num + 1] == self.eos_idx).clone().detach().bool()
            self.active_mask[eos_mask] = 0  # deactivate the newly ended cands
            self.lens[eos_mask] = step_num + 2  # deactivate the newly ended cands

        # Short circuit beam search if highest scoring candidate has ended
        top_cands = torch.argmax(self.scores, dim=1)  # (N, )
        finished_beam_mask = self.active_mask[torch.arange(self.N), top_cands] == 0  # (N, )  beam finished if top cand inactive
        self.active_mask[finished_beam_mask.nonzero().squeeze(1)] = 0  # set all cands in finished beams to inactive


class BeamSearch():
    def __init__(self, max_len, pad_idx, sos_idx, eos_idx, batch_size, beam_width):
        """
        TODO
        """
        self.T = max_len
        self.N = batch_size
        self.K = beam_width
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.rnn_hidden_states = None

        # sum of log probability scores
        self.scores = None  # (N, K)
        # input token indices (i.e. <sos> is token 0)
        self.seqs = None  # (T, N, K)
        # sequence lengths
        self.lens = None  # (N, K)
        # mask of which beam candidates have not finished decoding
        self.active_mask = None  # (N, K)

    def step(self, logits, step_num=0):
        """
        Beam search step
        :param logits: (N * K, V)
        :param step_num: int
        """
        lprob = torch.nn.functional.log_softmax(logits, dim=-1)  # step == 0 ? (N, V) : (N * K, V)

        if step_num == 0:
            # Initialize beams
            self.N = len(logits)
            self.seqs = torch.full((self.T, self.N, self.K), fill_value=self.pad_idx, dtype=torch.long, device=logits.device)
            self.lens = torch.zeros((self.N, self.K), dtype=torch.long, device=logits.device)
            self.scores, self.seqs[step_num + 1] = lprob.topk(self.K, sorted=False)  # (N, K), (N, K)

            # Check if any candidates reached <eos> and update active_mask and lens
            self.active_mask = (self.seqs[step_num + 1] != self.eos_idx).clone().detach().bool()  # (N, K)
            self.lens[~self.active_mask] = step_num + 2
        else:
            # Add lprobs to scores
            lprob = lprob.reshape(self.N, self.K, -1)  # (N, K, V)
            all_cand_scores = self.scores.unsqueeze(-1) + lprob  # (N, K, V) broadcast scores over V

            # Loop over each of the N beams
            for i in range(self.N):
                k = self.active_mask[i].sum()  # active beam width
                if k > 0:
                    # Extend each seq in the beam by k candidates
                    cand_scores, tokens = all_cand_scores[i].topk(k, sorted=False)  # (K, k), (K, k)
                    cand_scores[~self.active_mask[i]] = float('-inf')  # ensure inactive candidates are not selected

                    # Prune down to top k beams for each instance in the batch
                    pruned_scores, flat_idxs = torch.flatten(cand_scores).topk(k, sorted=False)  # (k, ), (k, )
                    pruned_row_idxs = torch.floor_divide(flat_idxs, self.K)  # the cands to extend
                    pruned_col_idxs = torch.fmod(flat_idxs, k)  # the tokens to extend them with
                    pruned_tokens = tokens[pruned_row_idxs, pruned_col_idxs]

                    # Update scores and seqs (note the use of nonzero().squeeze(1), which turns masks into indices)
                    active_idxs = self.active_mask[i].nonzero().squeeze(1)
                    self.scores[i, active_idxs] = pruned_scores
                    self.seqs[:step_num + 1, i, active_idxs] = self.seqs[:step_num + 1, i, pruned_row_idxs]  # base seqs
                    self.seqs[step_num + 1, i, active_idxs] = pruned_tokens  # extend the base seqs with the selected tokens

            # Check if any candidates reached <eos> and update active_mask and lens
            eos_mask = (self.seqs[step_num + 1] == self.eos_idx).clone().detach().bool()
            self.active_mask[eos_mask] = 0  # deactivate the newly ended cands
            self.lens[eos_mask] = step_num + 2  # deactivate the newly ended cands

        # Short circuit beam search if highest scoring candidate has ended
        top_cands = torch.argmax(self.scores, dim=1)  # (N, )
        finished_beam_mask = self.active_mask[torch.arange(self.N), top_cands] == 0  # (N, )  beam finished if top cand inactive
        self.active_mask[finished_beam_mask.nonzero().squeeze(1)] = 0  # set all cands in finished beams to inactive
