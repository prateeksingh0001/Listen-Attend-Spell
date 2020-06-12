import numpy as np
import os
import argparse


class DataPreprocessor:
    def __init__(self, file_list, mode):

        self.mode = mode
        if self.mode not in ("char", "word"):
            raise NotImplementedError("Specify the correct mode")

        if self.mode == "char":
            self.dict, self.vocab_size = self.create_char_dict(file_list)
        else:
            self.dict, self.vocab_size = self.create_word_dict(file_list)

        self.token2int = {j: i for i, j in enumerate(self.dict)}
        self.int2token = {i: j for i, j in enumerate(self.dict)}

    def create_char_dict(self, file_list):
        character_list = []

        for file in file_list:
            text = np.load(file, allow_pickle=True)
            for sentence in text:
                for word in sentence:
                    character_list.extend(word.astype('U').tolist())

        character_list.extend(['<SOS>', '<EOS>', ' '])
        character_list = np.unique(character_list).tolist()
        character_list.insert(0, '<PAD>')

        return character_list, len(character_list)

    def create_word_dict(self, file_list):
        word_list = []

        for file in file_list:
            text = np.load(file, allow_pickle=True)
            for sentence in text:
                for word in sentence:
                    word_list.append(word.astype['U'])

        word_list.extend(['<SOS>', '<EOS>', ' '])
        word_list = np.unique(word_list).tolist()
        word_list.insert(0, '<PAD>')

        return word_list, len(word_list)

    def transform_file2int(self, file_list):
        filelist2int = []

        for files in file_list:
            files2int = []
            for sentence in files:
                sentence2int = [self.sos()]
                sentence = [word.astype('U') for word in sentence]
                if self.mode == "char":
                    sent_chars = " ".join(sentence)
                    sentence2int = [self.sos()] + [self.get_int(c) for c in sent_chars] + [self.eos()]
                elif self.mode == "word":
                    sentence2int = [self.sos()] + [self.get_int(w) for w in sentence] + [self.eos()]
                files2int.append(sentence2int)
            filelist2int.append(files2int)

        return filelist2int

    def save_vocab_file(self, save_path):
        with open(save_path, "w") as f:
            for token in self.dict:
                f.write("%s\n" % token)

    def eos(self):
        return self.token2int['<EOS>']

    def sos(self):
        return self.token2int['<SOS>']

    def pad(self):
        return self.token2int['<PAD>']

    def get_int(self, token):
        return self.token2int[token]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_transcripts")
    parser.add_argument("--dev_transcripts")
    parser.add_argument("--vocab_save_path")
    parser.add_argument("--output_dir")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    train_transcripts = np.load(args.train_transcripts, allow_pickle=True)
    dev_transcripts = np.load(args.dev_transcripts, allow_pickle=True)

    preprocessor = DataPreprocessor([args.train_transcripts, args.dev_transcripts], mode="char")
    preprocessor.save_vocab_file(os.path.join(args.vocab_save_path, "vocab.txt"))

    train_transcripts_int, dev_transcripts_int = preprocessor.transform_file2int([train_transcripts, dev_transcripts])

    np.save(os.path.join(args.output_dir, "train_transcripts_int.npy"),
            train_transcripts_int)
    np.save(os.path.join(args.output_dir, "dev_transcripts_int.npy"),
            dev_transcripts_int)


if __name__ == "__main__":
    main()
