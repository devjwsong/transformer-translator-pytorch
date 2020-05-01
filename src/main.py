from tqdm import tqdm
from constants import *
from custom_data import *
from transformer import *
from torch import nn
from sklearn import metrics
from tensorboardX import SummaryWriter

import torch
import sys, os
import numpy as np
import argparse
import datetime

summary = SummaryWriter(summary_path)

class Manager():
    def __init__(self):
        # Load dataloaders
        print("Loading dataloaders...")
        self.train_loader, self.valid_loader, self.test_loader = get_data_loader()

        # Load vocabs
        print("Loading vocabs...")
        self.src_i2w = {}
        self.trg_i2w = {}

        with open(f"{SP_DIR}/{src_model_prefix}.vocab") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            word = line.strip().split('\t')[0]
            self.src_i2w[i] = word

        with open(f"{SP_DIR}/{trg_model_prefix}.vocab") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            word = line.strip().split('\t')[0]
            self.trg_i2w[i] = word

        print(f"The size of src vocab is {len(self.src_i2w)} and that of trg vocab is {len(self.trg_i2w)}.")

        # Load transformer model
        print("Loading Transformer model...")
        self.model = Transformer(src_vocab_size=len(self.src_i2w), trg_vocab_size=len(self.trg_i2w)).to(device)

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
        self.model.train()

        best_loss = sys.float_info.max
        total_training_time = datetime.timedelta()
        for epoch in range(1, num_epochs+1):

            train_losses = []
            start_time = datetime.datetime.now()

            for i, batch in tqdm(enumerate(self.train_loader)):
                src_input, input_trg, output_trg, encoder_mask, decoder_mask = batch
                src_input, input_trg, output_trg, encoder_mask, decoder_mask = \
                    src_input.to(device), input_trg.to(device), output_trg.to(device),\
                    encoder_mask.to(device), decoder_mask.to(device)

                output = self.model(src_input, input_trg, encoder_mask, decoder_mask) # (B, L, vocab_size)

                output_trg_shape = output_trg.shape
                self.optim.zero_grad()
                loss = self.criterion(output.view(-1, sp_vocab_size), output_trg.view(output_trg_shape[0] * output_trg_shape[1]))

                loss.backward()
                self.optim.step()

                train_losses.append(loss.item())

            end_time = datetime.datetime.now()
            training_time = end_time - start_time
            minutes = training_time.seconds // 60
            seconds = training_time.seconds

            mean_train_loss = np.mean(train_losses)
            print(f"#################### Epoch: {epoch} ####################")
            print(f"Train loss: {mean_train_loss} || Training time: {minutes}mins {seconds}secs")

            summary.add_scalar('loss/train_loss', mean_train_loss, epoch)

            if mean_train_loss < best_loss:
                if not os.path.exists(ckpt_dir):
                    os.mkdir(ckpt_dir)
                torch.save(self.model.state_dict(), f"{ckpt_dir}/best_model.pth")
                print(f"Current best model is saved.")
                best_loss = mean_train_loss

            total_training_time += training_time

        hours = total_training_time.seconds // 3600
        minutes = total_training_time.seconds // 60
        seconds = total_training_time.seconds
        print(f"Training finished! || Total training time: {hours}hrs {minutes}mins {seconds}secs")


    def test(self, model_name):
        if not os.path.exists(f"{ckpt_dir}/{model_name}"):
            print(f"There is no model named {model_name}. Test aborted.")
            return

        print("Testing starts.")
        self.model.load_state_dict(torch.load(f"{ckpt_dir}/model_name"))
        self.model.eval()

        start_time = datetime.datetime.now()
        for batch in tqdm(self.test_loader):
            pass

        end_time = datetime.datetime.now()

        total_testing_time = end_time - start_time
        hours = total_testing_time.seconds // 3600
        minutes = total_testing_time.seconds // 60
        seconds = total_testing_time.seconds

        print(f"Testing finished! || Total testing time: {hours}hrs {minutes}mins {seconds}secs")


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
