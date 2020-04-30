from tqdm import tqdm
from constants import *
from custom_data import *
from transformer import *
from torch import nn
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
from tensorboardX import SummaryWriter

import torch
import sys, os
import numpy as np
import argparse

summary = SummaryWriter(summary_path)

class Manager():
    def __init__(self):
        # Load dataloaders
        print("Loading dataloaders...")
        self.train_loader, self.valid_loader, self.test_loader = get_data_loader()

        # Load vocabs
        print("Loading vocabs...")
        self.src_w2i = {}
        self.tar_w2i = {}

        with open(f"{SP_DIR}/{src_model_prefix}.vocab") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            word = line.strip().split('\t')[0]
            self.src_w2i[word] = i

        with open(f"{SP_DIR}/{tar_model_prefix}.vocab") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            word = line.strip().split('\t')[0]
            self.tar_w2i[word] = i

        print(f"The size of src vocab is {len(self.src_w2i)} and that of tar vocab is {len(self.tar_w2i)}.")

        # Load transformer model
        print("Loading Transformer model...")
        self.model = Transformer(src_vocab_size=len(self.src_w2i), tar_vocab_size=len(self.tar_w2i)).to(device)

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

        best_valid_loss = sys.float_info.max
        for epoch in range(1, num_epochs+1):
            self.model.train()

            train_losses = []
            train_accuracies = []

            for i, batch in tqdm(enumerate(self.train_loader)):
                src_input, tar_input, tar_output, encoder_mask, decoder_mask = batch
                src_input, tar_input, tar_output, encoder_mask, decoder_mask = \
                    src_input.to(device), tar_input.to(device), tar_output.to(device),\
                    encoder_mask.to(device), decoder_mask.to(device)

                output = self.model(src_input, tar_input, encoder_mask, decoder_mask) # (B, L, vocab_size)

                self.optim.zero_grad()
                loss = self.criterion(output.view(-1, sp_vocab_size), tar_output.view(batch_size * seq_len))

                loss.backward()
                self.optim.step()

                train_losses.append(loss.item())

                output_list = torch.argmax(output, dim=-1).tolist()
                tar_output_list = tar_output.tolist()

                trimmed_output_list, trimmed_tar_output_list = self.trim_output(output_list, tar_output_list)

                train_accuracy = metrics.accuracy_score(trimmed_tar_output_list, trimmed_output_list)
                train_accuracies.append(train_accuracy)

            mean_train_loss = np.mean(train_losses)
            mean_train_accuracy = np.eman(train_accuracies)
            print(f"Epoch: {epoch}||Train loss: {mean_train_loss}||Train accuracy: {mean_train_accuracy}")

            summary.add_scalar('loss/train_loss', mean_train_loss, epoch)
            summary.add_scalar('accuracy/train_accuracy', train_accuracy, epoch)

            valid_loss, valid_accuracy = self.validation()

            summary.add_scalar('loss/valid_loss', valid_loss, epoch)
            summary.add_scalar('accuracy/valid_accuracy', valid_accuracy, epoch)

            if valid_loss < best_valid_loss:
                if not os.path.exists(ckpt_dir):
                    os.mkdir(ckpt_dir)
                torch.save(self.model.state_dict(), f"{ckpt_dir}/best_model.pth")
                print(f"Current best model is saved.")
                best_valid_loss = valid_loss

            print(f"Best validation loss: {best_valid_loss}||Validation loss: {valid_loss}||Valid accuracy: {valid_accuracy}")

        print(f"Training finished!")


    def validation(self):
        self.model.eval()

        valid_losses = []
        valid_accuracies = []

        for batch in tqdm(self.valid_loader):
            src_input, _, tar_output, encoder_mask, _ = batch
            src_input, tar_output, encoder_mask = \
                src_input.to(device), tar_output.to(device), encoder_mask.to(device)

            tar_input = torch.zeros(tar_output.shape[0], tar_output.shape[1]).long()
            for seq in tar_input:
                seq[0] = sos_id

            output = self.model(src_input, tar_input, encoder_mask)  # (B, L, vocab_size)
            loss = self.criterion(output.view(-1, sp_vocab_size), tar_output.view(batch_size * seq_len))

            valid_losses.append(loss.item())

            output_list = torch.argmax(output, dim=-1).tolist()
            tar_output_list = tar_output.tolist()

            trimmed_output_list, trimmed_tar_output_list = self.trim_output(output_list, tar_output_list)

            valid_accuracy = metrics.accuracy_score(trimmed_tar_output_list, trimmed_output_list)
            valid_accuracies.append(valid_accuracy)

        mean_valid_loss = np.mean(valid_losses)
        mean_valid_accuracy = np.mean(valid_accuracies)

        return mean_valid_loss, mean_valid_accuracy


    def test(self, model_name):
        if not os.path.exists(f"{ckpt_dir}/{model_name}"):
            print(f"There is no model named {model_name}. Test aborted.")
            return

        print("Testing starts.")
        self.model.load_state_dict(torch.load(f"{ckpt_dir}/model_name"))
        self.model.eval()

        test_accuracies = []

        for batch in tqdm(self.test_loader):
            src_input, _, tar_output, encoder_mask, _ = batch
            src_input, tar_output, encoder_mask = \
                src_input.to(device), tar_output.to(device), encoder_mask.to(device)

            tar_input = torch.zeros(tar_output.shape[0], tar_output.shape[1]).long()
            for seq in tar_input:
                seq[0] = sos_id

            output = self.model(src_input, tar_input, encoder_mask)  # (B, L, vocab_size)
            output = torch.argmax(output, dim=-1) # (B, L)

            output_list = output.tolist()
            tar_output_list = tar_output.tolist()

            trimmed_output_list, trimmed_tar_output_list = self.trim_output(output_list, tar_output_list)

            test_accuracy = metrics.accuracy_score(trimmed_tar_output_list, trimmed_output_list)
            test_accuracies.append(test_accuracy)

        mean_test_accuracy = np.mean(test_accuracies)

        print(f"Testing finished! Test accuracy: {mean_test_accuracy}")


    def trim_output(self, output_list, tar_output_list):
        trimmed_output_list = []
        trimmed_tar_output_list = []
        for i in range(len(output_list)):
            tar_output = [idx for idx in tar_output_list[i] if idx != pad_id]
            original_len = len(tar_output)
            output = [idx for j, idx in enumerate(output_list[i]) if j<original_len]

            trimmed_output_list += output
            trimmed_tar_output_list += tar_output

        return trimmed_output_list, trimmed_tar_output_list


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
