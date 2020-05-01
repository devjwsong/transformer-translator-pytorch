from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from constants import *

import torch
import sentencepiece as spm
import numpy as np

src_sp = spm.SentencePieceProcessor()
trg_sp = spm.SentencePieceProcessor()
src_sp.Load(f"{SP_DIR}/{src_model_prefix}.model")
trg_sp.Load(f"{SP_DIR}/{trg_model_prefix}.model")


def get_data_loader():
    with open(f"{DATA_DIR}/{SRC_DATA_NAME}", 'r') as f:
        src_text_list = f.readlines()

    with open(f"{DATA_DIR}/{trg_DATA_NAME}", 'r') as f:
        trg_text_list = f.readlines()

    print("Tokenizing & Padding src data...")
    src_list = process_src(src_text_list) # (sample_num, L)
    print(f"The shape of src data: {np.shape(src_list)}")

    print("Tokenizing & Padding trg data...")
    input_trg_list, output_trg_list = process_trg(trg_text_list) # (sample_num, L)
    print(f"The shape of input trg data: {np.shape(input_trg_list)}")
    print(f"The shape of output trg data: {np.shape(output_trg_list)}")

    dataset = CustomDataset(src_list, input_trg_list, output_trg_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def add_padding(tokenized_text):
    if len(tokenized_text) < seq_len:
        left = seq_len - len(tokenized_text)
        padding = [pad_id] * left
        tokenized_text += padding

    return tokenized_text


def process_src(text_list):
    tokenized_list = []
    for text in tqdm(text_list):
        tokenized = src_sp.EncodeAsIds(text.strip())
        tokenized_list.append(add_padding(tokenized))

    return tokenized_list

def process_trg(text_list):
    input_tokenized_list = []
    output_tokenized_list = []
    for text in tqdm(text_list):
        tokenized = trg_sp.EncodeAsIds(text.strip())
        trg_input = [sos_id] + tokenized[:-1]
        trg_output = tokenized[1:] + [eos_id]
        input_tokenized_list.append(add_padding(trg_input))
        output_tokenized_list.append(add_padding(trg_output))

    return input_tokenized_list, output_tokenized_list


class CustomDataset(Dataset):
    def __init__(self, src_list, input_trg_list, output_trg_list):
        super().__init__()
        self.src_data = torch.LongTensor(src_list)
        self.input_trg_data = torch.LongTensor(input_trg_list)
        self.output_trg_data = torch.LongTensor(output_trg_list)

        assert np.shape(src_list) == np.shape(input_trg_list), "The shape of src_list and input_trg_list are different."
        assert np.shape(input_trg_list) == np.shape(output_trg_list), "The shape of input_trg_list and output_trg_list are different."

        self.encoder_mask, self.decoder_mask = self.make_mask()

    def make_mask(self):
        encoder_mask = (self.src_data != pad_id).unsqueeze(1) # (num_samples, 1, L)
        decoder_mask = (self.input_trg_data != pad_id).unsqueeze(1) # (num_samples, 1, L)

        nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool) # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask) # (1, L, L) to triangular shape
        decoder_mask = decoder_mask & nopeak_mask # (num_samples, L, L) padding false

        return encoder_mask, decoder_mask

    def __getitem__(self, idx):
        return self.src_data[idx], self.input_trg_data[idx], self.output_trg_data[idx], \
        self.encoder_mask[idx], self.decoder_mask[idx]

    def __len__(self):
        return np.shape(self.src_data)[0]
