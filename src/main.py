from tqdm import tqdm
from constants import *
from custom_data import *
from transformer import *
from torch import nn
from torchtext.data import metrics

import torch
import sys, os
import numpy as np
import argparse
import sentencepiece as spm


class Manager():
    def __init__(self):
        # Load dataloaders
        print("Loading dataloaders...")
        self.train_loader, self.valid_loader, self.test_loader = get_data_loader()

        # Load vocabs
        print("Loading vocabs...")
        self.src_i2v = {}
        self.tar_i2v = {}

        with open(f"{SP_DIR}/{src_model_prefix}.vocab") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            word = line.strip().split('\t')[0]
            self.src_i2v[i] = word

        with open(f"{SP_DIR}/{tar_model_prefix}.vocab") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            word = line.strip().split('\t')[0]
            self.tar_i2v[i] = word

        print(f"The size of src vocab is {len(self.src_i2v)} and that of tar vocab is {len(self.tar_i2v)}.")

        # Load transformer model
        print("Loading Transformer model...")
        self.model = Transformer(src_vocab_size=len(self.src_i2v), tar_vocab_size=len(self.tar_i2v)).to(device)

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Load optimizer and loss function
        print("Loading optimizer and loss function...")
        self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.NLLLoss(ignore_index=pad_id)

        print("Setting finished.")

    def train(self):
        print("Training starts.")
        for epoch in range(1, num_epochs+1):
            self.model.train()

            train_losses = []
            train_bleu_scores = []
            best_valid_loss = sys.float_info.max

            for batch in tqdm(self.train_loader):
                src_input, tar_input, tar_output, encoder_mask, masked_attn_mask, attn_mask = batch
                src_input, tar_input, tar_output, encoder_mask, masked_attn_mask, attn_mask = \
                    src_input.to(device), tar_input.to(device), tar_output.to(device),\
                    encoder_mask.to(device), masked_attn_mask.to(device), attn_mask.to(device)

                output = self.model(src_input, tar_input, encoder_mask, masked_attn_mask, attn_mask) # (B, L, vocab_size)

                self.optim.zero_grad()
                loss = self.criterion(output.view(-1, sp_vocab_size), tar_output.view(batch_size * seq_len))

                loss.backward()
                self.optim.step()

                train_losses.append(loss.item())

                output_list = torch.argmax(output, dim=-1).tolist()
                tar_output_list = tar_output.tolist()

                decoded_output_list, decoded_tar_output_list = self.decode_tokens(output_list, tar_output_list)

                train_bleu_score = metrics.bleu_score(decoded_output_list, decoded_tar_output_list, max_n=4)
                train_bleu_scores.append(train_bleu_score)

            mean_train_loss = np.mean(train_losses)
            mean_bleu_score = np.mean(train_bleu_scores)
            print(f"Epoch: {epoch}||Train loss: {mean_train_loss}||Train BLEU score: {mean_bleu_score}")

            valid_loss, valid_bleu_score = self.validation()

            if valid_loss < best_valid_loss:
                if not os.path.exists(ckpt_dir):
                    os.mkdir(ckpt_dir)
                torch.save(self.model.state_dict(), f"{ckpt_dir}/best_model.pth")
                print(f"Current best model is saved.")

            print(f"Best validation loss: {best_valid_loss}||Validation loss: {valid_loss}||Valid BLEU score: {valid_bleu_score}")

        print(f"Training finished!")


    def validation(self):
        self.model.eval()

        valid_losses = []
        valid_bleu_scores = []
        for batch in tqdm(self.valid_loader):
            src_input, tar_input, tar_output, encoder_mask, masked_attn_mask, attn_mask = batch
            src_input, tar_input, tar_output, encoder_mask, masked_attn_mask, attn_mask = \
                src_input.to(device), tar_input.to(device), tar_output.to(device), \
                encoder_mask.to(device), masked_attn_mask.to(device), attn_mask.to(device)

            output = self.model(src_input, tar_input, encoder_mask, masked_attn_mask, attn_mask)  # (B, L, vocab_size)
            loss = self.criterion(output.view(-1, sp_vocab_size), tar_output.view(batch_size * seq_len))

            valid_losses.append(loss.item())

            output_list = torch.argmax(output, dim=-1).tolist()
            tar_output_list = tar_output.tolist()

            decoded_output_list, decoded_tar_output_list = self.decode_tokens(output_list, tar_output_list)

            valid_bleu_score = metrics.bleu_score(decoded_output_list, decoded_tar_output_list, max_n=4)
            valid_bleu_scores.append(valid_bleu_score)

        mean_valid_loss = np.mean(valid_losses)
        mean_bleu_score = np.mean(valid_bleu_scores)

        return mean_valid_loss, mean_bleu_score


    def test(self, model_name):
        if not os.path.exists(f"{ckpt_dir}/{model_name}"):
            print(f"There is no model named {model_name}. Test aborted.")
            return

        print("Testing starts.")
        self.model.load_state_dict(torch.load(f"{ckpt_dir}/model_name"))
        self.model.eval()

        test_bleu_scores = []
        for batch in tqdm(self.test_loader):
            src_input, _, tar_output, encoder_mask, masked_attn_mask, attn_mask = batch
            src_input, tar_output, encoder_mask, masked_attn_mask, attn_mask = \
                src_input.to(device), tar_output.to(device), \
                encoder_mask.to(device), masked_attn_mask.to(device), attn_mask.to(device)

            tar_input = torch.zeros(tar_output.shape[0], tar_output.shape[1]).long()
            for seq in tar_input:
                seq[0] = sos_id

            output = self.model(src_input, tar_input, encoder_mask, masked_attn_mask, attn_mask)  # (B, L, vocab_size)
            output = torch.argmax(output, dim=-1) # (B, L)

            output_list = output.tolist()
            tar_output_list = tar_output.tolist()

            decoded_output_list, decoded_tar_output_list = self.decode_tokens(output_list, tar_output_list)

            test_bleu_score = metrics.bleu_score(decoded_output_list, decoded_tar_output_list, max_n=4)
            test_bleu_scores.append(test_bleu_score)

        mean_bleu_score = np.mean(test_bleu_scores)

        print(f"Testing finished! Test BLEU score: {mean_bleu_score}")


    def decode_tokens(self, output_list, tar_output_list):
        decoded_output_list = []
        decoded_tar_output_list = []
        for i in range(len(output_list)):
            tar_output = [idx for idx in tar_output_list[i] if idx != pad_id]
            original_len = len(tar_output)
            output = [idx for i, idx in enumerate(output_list[i]) if i<original_len]

            decoded_output = [self.tar_i2v[idx] for idx in output]
            decoded_tar_output = [self.tar_i2v[idx] for idx in tar_output]

            decoded_output_list.append(decoded_output)
            decoded_tar_output_list.append(decoded_tar_output)

        return decoded_output_list, decoded_tar_output_list



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, help="train or test?")
    parser.add_argument('--model_name', required=False, help="trained model file to test")

    args = parser.parse_args()

    manager = Manager()

    if args.mode == 'train':
        manager.train()
    elif args.mode == 'test':
        if args.model_name is None:
            print("Please specify the model file.")
        else:
            manager.test(args.model_name)
    else:
        print("Please specify mode argument either with 'train' or 'test'.")
