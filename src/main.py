from tqdm import tqdm
from src.constants import *
from src.custom_data import *
from src.transformer import *
from torch import nn
from torchtext.data import metrics

import torch
import sys, os
import numpy as np


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
            total_loss = 0.0
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

                total_loss += loss.item()

            print(f"Epoch: {epoch}||Train loss: {total_loss}")

            valid_loss = self.validation()

            if valid_loss < best_valid_loss:
                if not os.path.exists(ckpt_dir):
                    os.mkdir(ckpt_dir)
                torch.save(self.model.state_dict(), f"{ckpt_dir}/best_model.pth")
                print(f"Current best model is saved.")

            print(f"Current validation loss: {valid_loss}||Best validation loss: {best_valid_loss}")

        print(f"Training finished!")


    def validation(self):
        self.model.eval()

        total_loss = 0.0
        for batch in tqdm(self.valid_loader):
            src_input, tar_input, tar_output, encoder_mask, masked_attn_mask, attn_mask = batch
            src_input, tar_input, tar_output, encoder_mask, masked_attn_mask, attn_mask = \
                src_input.to(device), tar_input.to(device), tar_output.to(device), \
                encoder_mask.to(device), masked_attn_mask.to(device), attn_mask.to(device)

            output = self.model(src_input, tar_input, encoder_mask, masked_attn_mask, attn_mask)  # (B, L, vocab_size)
            loss = self.criterion(output.view(-1, sp_vocab_size), tar_output.view(batch_size * seq_len))

            total_loss += loss.item()

        return total_loss


    def test(self):
        self.model.eval()

        blue_scores = []
        for batch in tqdm(self.test_loader):
            src_input, tar_input, tar_output, encoder_mask, masked_attn_mask, attn_mask = batch
            src_input, tar_input, tar_output, encoder_mask, masked_attn_mask, attn_mask = \
                src_input.to(device), tar_input.to(device), tar_output.to(device), \
                encoder_mask.to(device), masked_attn_mask.to(device), attn_mask.to(device)

            output = self.model(src_input, tar_input, encoder_mask, masked_attn_mask, attn_mask)  # (B, L, vocab_size)
            output = torch.argmax(output, dim=-1) # (B, L)

            blue_score = metrics.bleu_score(output, tar_output, max_n=4)
            blue_scores.append(blue_score)

        mean_blue_score = np.mean(blue_scores)

        print(f"Testing finished! Mean of test BLUE score: {mean_blue_score}")
