import torch

# Path or parameters for data
DATA_DIR = '../data'
SP_DIR = '../data/sp'
SRC_DATA_NAME = 'full_data.en'
TRG_DATA_NAME = 'full_data.fr'

# Parameters for sentencepiece tokenizer
pad_id = 0
sos_id = 1
eos_id = 2
unk_id = 3
src_model_prefix = 'src_sp'
trg_model_prefix = 'trg_sp'
sp_vocab_size = 16000
character_coverage = 1.0
model_type = 'bpe'

# Parameters for Transformer & training
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
learning_rate = 0.0001
batch_size = 16
seq_len = 320
num_heads = 8
num_layers = 6
d_model = 512
d_ff = 2048
d_k = d_model // num_heads
drop_out_rate = 0.1
num_epochs = 15
ckpt_dir = '../saved_model'

# Path for tensorboard
summary_path = '../runs'
