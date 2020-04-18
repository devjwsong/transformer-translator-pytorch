from tqdm import tqdm
from src.constants import *
from src.custom_data import *
from src.transformer import *
from torch import nn
from torchtext.data import metrics

import torch
import sys, os
import numpy as np
import argparse


class Manager():
    def __init__(self):
        # Load dataloaders
        self.train_loader, self.valid_loader, self.test_loader = get_data_loader()

        # Load transformer model
        self.model = Transformer(src_vocab_size=sp_vocab_size, tar_vocab_size=sp_vocab_size).to(device)

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Load optimizer and loss function
        self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.NLLLoss(ignore_index=pad_id)

    def train(self):
        for epoch in range(1, num_epochs):
            self.model.train()

            train_losses = []
            train_blue_scores = []
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

                train_blue_score = metrics.bleu_score(torch.argmax(output, dim=-1), tar_output, max_n=4)
                train_blue_scores.append(train_blue_score)

            mean_train_loss = np.mean(train_losses)
            mean_blue_score = np.mean(train_blue_scores)
            print(f"Epoch: {epoch}||Train loss: {mean_train_loss}||Train BLUE score: {mean_blue_score}")

            valid_loss, valid_blue_score = self.validation()

            if valid_loss < best_valid_loss:
                if not os.path.exists(ckpt_dir):
                    os.mkdir(ckpt_dir)
                torch.save(self.model.state_dict(), f"{ckpt_dir}/best_model.pth")
                print(f"Current best model is saved.")

            print(f"Best validation loss: {best_valid_loss}||Validation loss: {valid_loss}||Valid BLUE score: {valid_blue_score}")

        print(f"Training finished!")


    def validation(self):
        self.model.eval()

        valid_losses = []
        valid_blue_scores = []
        for batch in tqdm(self.valid_loader):
            src_input, tar_input, tar_output, encoder_mask, masked_attn_mask, attn_mask = batch
            src_input, tar_input, tar_output, encoder_mask, masked_attn_mask, attn_mask = \
                src_input.to(device), tar_input.to(device), tar_output.to(device), \
                encoder_mask.to(device), masked_attn_mask.to(device), attn_mask.to(device)

            output = self.model(src_input, tar_input, encoder_mask, masked_attn_mask, attn_mask)  # (B, L, vocab_size)
            loss = self.criterion(output.view(-1, sp_vocab_size), tar_output.view(batch_size * seq_len))

            valid_losses.append(loss.item())

            valid_blue_score = metrics.bleu_score(torch.argmax(output, dim=-1), tar_output, max_n=4)
            valid_blue_scores.append(valid_blue_score)

        mean_valid_loss = np.mean(valid_losses)
        mean_blue_score = np.mean(valid_blue_scores)

        return mean_valid_loss, mean_blue_score


    def test(self, model_name):
        if not os.path.exists(f"{ckpt_dir}/{model_name}"):
            print(f"There is no model named {model_name}. Test aborted.")
            return

        self.model.load_state_dict(torch.load(f"{ckpt_dir}/model_name"))
        self.model.eval()

        test_blue_scores = []
        for batch in tqdm(self.test_loader):
            src_input, tar_input, tar_output, encoder_mask, masked_attn_mask, attn_mask = batch
            src_input, tar_input, tar_output, encoder_mask, masked_attn_mask, attn_mask = \
                src_input.to(device), tar_input.to(device), tar_output.to(device), \
                encoder_mask.to(device), masked_attn_mask.to(device), attn_mask.to(device)

            output = self.model(src_input, tar_input, encoder_mask, masked_attn_mask, attn_mask)  # (B, L, vocab_size)
            output = torch.argmax(output, dim=-1) # (B, L)

            test_blue_score = metrics.bleu_score(output, tar_output, max_n=4)
            test_blue_scores.append(test_blue_score)

        mean_blue_score = np.mean(test_blue_scores)

        print(f"Testing finished! Test BLUE score: {mean_blue_score}")


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
        print("Please specify mode argument with 'train' or 'test'.")
