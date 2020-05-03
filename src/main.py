from tqdm import tqdm
from constants import *
from custom_data import *
from transformer import *
from torch import nn
from tensorboardX import SummaryWriter

import torch
import sys, os
import numpy as np
import argparse
import datetime
import sentencepiece as spm

summary = SummaryWriter(summary_path)

class Manager():
    def __init__(self, is_train=True):
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

        if is_train:
            for p in self.model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

            # Load optimizer and loss function
            print("Loading optimizer and loss function...")
            self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            self.criterion = nn.NLLLoss(ignore_index=pad_id)

            # Load dataloaders
            print("Loading dataloaders...")
            self.train_loader = get_data_loader()

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
            seconds = training_time.seconds % 60

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

        seconds = total_training_time.seconds
        hours = seconds // 3600
        seconds = seconds % 3600
        minutes = seconds // 60
        seconds = seconds % 60
        print(f"Training finished! || Total training time: {hours}hrs {minutes}mins {seconds}secs")


    def test(self, model_name, input_sentence):
        if not os.path.exists(f"{ckpt_dir}/{model_name}"):
            print(f"There is no model named {model_name}. Test aborted.")
            return

        print("Testing starts.")
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(f"{ckpt_dir}/{model_name}"))
        else:
            self.model.load_state_dict(torch.load(f"{ckpt_dir}/{model_name}", map_location=torch.device('cpu')))
        self.model.eval()

        print("Loading sentencepiece tokenizer...")
        src_sp = spm.SentencePieceProcessor()
        trg_sp = spm.SentencePieceProcessor()
        src_sp.Load(f"{SP_DIR}/{src_model_prefix}.model")
        trg_sp.Load(f"{SP_DIR}/{trg_model_prefix}.model")

        print("Preprocessing input sentence...")
        tokenized = src_sp.EncodeAsIds(input_sentence)
        src_data = torch.LongTensor(add_padding(tokenized)).unsqueeze(0).to(device) # (1, L)
        encoder_mask = (src_data != pad_id).unsqueeze(1).to(device) # (1, 1, L)

        start_time = datetime.datetime.now()

        print("Encoding input sentence...")
        src_embedded = self.model.src_embedding(src_data)
        src_positional_encoded = self.model.positional_encoder(src_embedded)
        encoder_output = self.model.encoder(src_positional_encoded, encoder_mask) # (1, L, d_model)

        outputs = torch.zeros(seq_len).long().to(device) # (L)
        outputs[0] = sos_id # (L)
        output_len = 0

        for i in range(1, seq_len):
            decoder_mask = (outputs.unsqueeze(0) != pad_id).unsqueeze(1).to(device) # (1, 1, L)
            nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool).to(device)  # (1, L, L)
            nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L) to triangular shape
            decoder_mask = decoder_mask & nopeak_mask  # (1, L, L) padding false

            trg_embedded = self.model.trg_embedding(outputs.unsqueeze(0))
            trg_positional_encoded = self.model.positional_encoder(trg_embedded)
            decoder_output = self.model.decoder(trg_positional_encoded, encoder_output, encoder_mask, decoder_mask) # (1, L, d_model)

            output = self.model.softmax(self.model.output_linear(decoder_output)) # (1, L, trg_vocab_size)
            output = torch.argmax(output, dim=-1) # (1, L)
            last_word_id = output[0][-1]

            if last_word_id == eos_id:
                break

            outputs[i] = last_word_id
            output_len = i

        decoded_output = outputs[1:][:output_len].tolist()
        decoded_output = trg_sp.decode_ids(decoded_output)

        end_time = datetime.datetime.now()

        total_testing_time = end_time - start_time
        seconds = total_testing_time.seconds
        minutes = seconds // 60
        seconds = seconds % 60

        print(f"Result: {decoded_output}")
        print(f"Testing finished! || Total testing time: {minutes}mins {seconds}secs")


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, help="train or test?")
    parser.add_argument('--model_name', required=False, help="trained model file to test")
    parser.add_argument('--input', type=str, required=False, help="input sentence when testing")

    args = parser.parse_args()

    if args.mode == 'train':
        manager = Manager(is_train=True)
        manager.train()
    elif args.mode == 'test':
        if args.model_name is None:
            print("Please specify the model file.")
        else:
            if args.input is None:
                print("Please input a source sentence.")
            else:
                manager = Manager(is_train=False)
                manager.test(args.model_name, args.input)
    else:
        print("Please specify mode argument either with 'train' or 'test'.")
