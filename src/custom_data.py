from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from .constants import *

import torch
import sentencepiece as spm
import numpy as np

src_sp = spm.SentencePieceProcessor()
tar_sp = spm.SentencePieceProcessor()
src_sp.Load(f"{SP_DIR}/{src_model_prefix}.model")
tar_sp.Load(f"{SP_DIR}/{tar_model_prefix}.model")

train_frac = 0.8
valid_test_split_frac = 0.1


def get_data_loader():
    with open(f"{DATA_DIR}/{SRC_DATA_NAME}", 'r') as f:
        src_text_list = f.readlines()

    with open(f"{DATA_DIR}/{TAR_DATA_NAME}", 'r') as f:
        tar_text_list = f.readlines()

    src__list = process_src(src_text_list) # (sample_num, L)
    tar_input_list, tar_output_list = process_tar(tar_text_list) # (sample_num, L)

    src_train_list, src_valid_list, src_test_list = split_data(src__list)
    tar_input_train_list, tar_input_valid_list, tar_input_test_list = split_data(tar_input_list)
    tar_output_train_list, tar_output_valid_list, tar_output_test_list = split_data(tar_output_list)

    train_dataset = CustomDataset(src_train_list, tar_input_train_list, tar_output_train_list)
    valid_dataset = CustomDataset(src_valid_list, tar_input_valid_list, tar_output_valid_list)
    test_dataset = CustomDataset(src_test_list, tar_input_test_list, tar_output_test_list)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader


def split_data(data_list):
    train_list = data_list[:int(len(data_list) * train_frac)]
    remained_list = data_list[int(len(data_list) * train_frac):]

    valid_list = remained_list[:int(len(remained_list) * valid_test_split_frac)]
    test_list = remained_list[int(len(remained_list) * valid_test_split_frac):]

    return train_list, valid_list, test_list


def add_padding(tokenized_text):
    if len(tokenized_text) < seq_len:
        left = seq_len - len(tokenized_text)
        padding = [pad_id] * left
        tokenized_text += padding

    return tokenized_text


def process_src(text_list):
    print("Tokenizing & Padding src data...")
    tokenized_list = []
    for text in tqdm(text_list):
        tokenized = src_sp.EncodeAsIds(text.strip())
        tokenized_list.append(add_padding(tokenized))

    print(f"The shape of src data: {np.shape(tokenized_list)}")
    return tokenized_list


def process_tar(text_list):
    print("Tokenizing & Padding tar data...")
    input_list = []
    output_list = []
    for text in tqdm(text_list):
        tokenized = tar_sp.EncodeAsIds(text.strip())
        input_tokenized = [sos_id] + tokenized
        output_tokenized = tokenized + [eos_id]
        input_list.append(add_padding(input_tokenized))
        output_list.append(add_padding(output_tokenized))

    print(f"The shape of tar(input) data: {np.shape(input_list)}")
    print(f"The shape of tar(output) data: {np.shape(output_list)}")
    return input_list, output_list


class CustomDataset(Dataset):
    def __init__(self, src_list, tar_input_list, tar_output_list):
        super().__init__()
        self.src_data = torch.FloatTensor(src_list)
        self.tar_input_data = torch.FloatTensor(tar_input_list)
        self.tar_output_data = torch.FloatTensor(tar_output_list)

        assert np.shape(src_list) == np.shape(tar_input_list), "The shape of src_list and tar_input_list are different."
        assert np.shape(tar_input_list) == np.shape(tar_output_list), "The shape of tar_input_list and tar_output_list are different."

        self.encoder_mask, self.masked_attn_mask, self.attn_mask = self.make_mask()

    def make_mask(self):
        encoder_mask = (self.src_data != pad_id).unsqueeze(1) # (num_samples, 1, L)
        attn_mask = (self.tar_input_data != pad_id).unsqueeze(1) # (num_samples, 1, L)

        masked_attn_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool) # (1, L, L)
        masked_attn_mask = torch.tril(masked_attn_mask) # (1, L, L) to triangular shape
        masked_attn_mask = attn_mask & masked_attn_mask # (num_samples, L, L) padding false

        return encoder_mask, masked_attn_mask, attn_mask

    def __getitem__(self, idx):
        return self.src_data[idx], self.tar_input_data[idx], self.tar_output_data[idx], \
               self.encoder_mask[idx], self.masked_attn_mask[idx], self.attn_mask[idx]

    def __len__(self):
        return np.shape(self.src_list)[0]
