import os
import numpy as np
import argparse

# Character based vocab builder
def make_vocab(files):

    characters = []
    characters.append("<SOS>")
    characters.append("<EOS>")
    characters.append(" ")

    for file in files:
        for line in file:
            for word in line:
                characters.extend(list(word.decode("utf-8")))

    vocab_list = np.unique(characters).tolist()
    vocab_list.insert(0, "<PAD>")
    return vocab_list

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_transcripts")
    parser.add_argument("--dev_transcripts")
    parser.add_argument("--output_dir")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    train_transcripts = np.load(args.train_transcripts, allow_pickle=True)
    dev_transcripts = np.load(args.dev_transcripts, allow_pickle=True)

    vocab_list = make_vocab([train_transcripts, dev_transcripts])
    # vocab_list = [i.decode("utf-8") for i in vocab_list]

    print("vocab size = ", len(vocab_list))
    print("SOS index = ", vocab_list.index("<SOS>"))
    print("EOS index = ", vocab_list.index("<EOS>"))
    print("PAD index = ", vocab_list.index("<PAD>"))

    with open(os.path.join(args.output_dir, "vocab.txt"), "w") as file:
        file.write('\n'.join(vocab_list))

if __name__ == "__main__":
    main()
