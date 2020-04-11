from config import *

import os
import sentencepiece as spm

def train_sp(is_src=True):
    template = "--input={} \
                --pad_id={} \
                --bos_id={} \
                --eos_id={} \
                --unk_id={} \
                --model_prefix={} \
                --vocab_size={} \
                --character_coverage={} \
                --model_type={}"

    if is_src:
        this_input_file = f"{DATA_DIR}/{SRC_DATA_NAME}"
        this_model_prefix = f"{SP_DIR}/{src_model_prefix}"
    else:
        this_input_file = f"{DATA_DIR}/{TAR_DATA_NAME}"
        this_model_prefix = f"{SP_DIR}/{tar_model_prefix}"

    config = template.format(this_input_file,
                            pad_id,
                            sos_id,
                            eos_id,
                            unk_id,
                            this_model_prefix,
                            vocab_size,
                            character_coverage,
                            model_type)

    print(config)

    if not os.path.isdir(SP_DIR):
        os.mkdir(SP_DIR)

    print(spm)
    spm.SentencePieceTrainer.Train(config)


if __name__=='__main__':
    train_sp(is_src=True)
    train_sp(is_src=False)
