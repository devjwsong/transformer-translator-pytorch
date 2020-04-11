from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from config import *

import torch
import sentencepiece as spm
import numpy as np

src_sp = spm.SentencePieceProcessor()
tar_sp = spm.SentencePieceProcessor()
src_sp.Load(f"{SP_DIR}/{src_model_prefix}.model")
tar_sp.Load(f"{SP_DIR}/{tar_model_prefix}.model")

train_frac = 0.8


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
        self.src_list = src_list
        self.tar_input_list = tar_input_list
        self.tar_output_list = tar_output_list

        assert np.shape(src_list) == np.shape(tar_input_list), "The shape of src_list and tar_input_list are different."
        assert np.shape(tar_input_list) == np.shape(tar_output_list), "The shape of tar_input_list and tar_output_list are different."

    def __getitem__(self, idx):
        src_data = torch.FloatTensor(self.src_list[idx])
        tar_input_data = torch.FloatTensor(self.tar_input_list[idx])
        tar_output_data = torch.FloatTensor(self.tar_output_list[idx])

        return src_data, tar_input_data, tar_output_data

    def __len__(self):
        return np.shape(self.src_list)[0]
