# transformer-translator-pytorch
This is a machine translation project using the basic **Transformer** introduced in *Attention is all you need*[[1]](#1).

I used English-French corpus provided by "European Parliament Proceedings Parallel Corpus 1996-2011"[[2]](#2).
(You can use any other datasets, of course.)

<br/>

---

### Configurations

You can set various hyperparameters in `src/constants.py` file.

The description of each variable is as follows.

<br/>

**Parameters for data**

Argument | Type | Description | Default
---------|------|---------------|------------
`DATA_DIR` | `str` | Name of the parent directory where data files are stored. | `'data'` 
`SP_DIR` | `str` | Path for the directory which contains the sentence tokenizers and vocab files. | `f'{DATA_DIR}/sp'` 
`SRC_DIR` | `str` | Name of the directory which contains the source data files. | `'src'` 
`TRG_DIR` | `str` | Name of the directory which contains the target data files. | `'trg'` 
`SRC_RAW_DATA_NAME` | `str` | Name of the source raw data file. | `raw_data.src` 
`TRG_RAW_DATA_NAME` | `str` | Name of the target raw data file. | `raw_data.trg` 
`TRAIN_NAME` | `str` | Name of the train data file. | `train.txt` 
`VALID_NAME` | `str` | Name of the validation data file. | `valid.txt` 
`TEST_NAME` | `str` | Name of the test data file. | `test.txt` 

<br/>

**Parameters for Sentencepiece**

| Argument             | Type    | Description                                                  | Default   |
| -------------------- | ------- | ------------------------------------------------------------ | --------- |
| `pad_id`             | `int`   | The index of pad token.                                      | `0`       |
| `sos_id`             | `int`   | The index of start token.                                    | `1`       |
| `eos_id`             | `int`   | The index of end token.                                      | `2`       |
| `unk_id`             | `int`   | The index of unknown token.                                  | `3`       |
| `src_model_prefix`   | `str`   | The file name prefix for the source language tokenizer & vocabulary. | `src_sp`  |
| `trg_model_prefix`   | `str`   | The file name prefix for the target language tokenizer & vocabulary. | `trg_sp`  |
| `sp_vocab_size`      | `int`   | The size of vocabulary.                                      | `16000`   |
| `character_coverage` | `float` | The value for character coverage.                            | `1.0`     |
| `model_type`         | `str`   | The type of sentencepiece model. (`unigram`, `bpe`, `char`, or `word`) | `unigram` |

<br/>

**Parameters for the transformer & training**

| Argument        | Type           | Description                                                  | Default                                                      |
| --------------- | -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `device`        | `torch.device` | The device type. (CUDA or CPU)                               | `torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')` |
| `learning_rate` | `float`        | The learning rate.                                           | `1e-4`                                                       |
| `batch_size`    | `int`          | The batch size.                                              | `80`                                                         |
| `seq_len`       | `int`          | The maximum length of a sentence.                            | `200`                                                        |
| `num_heads`     | `int`          | The number of heads for Multi-head attention.                | `8`                                                          |
| `num_layers`    | `int`          | The number of layers in the encoder & the decoder.           | `6`                                                          |
| `d_model`       | `int`          | The size of hidden states in the model.                      | `512`                                                        |
| `d_ff`          | `int`          | The size of intermediate  hidden states in the feed-forward layer. | `2048`                                                       |
| `d_k`           | `int`          | The size of dimension which a single head should take. (Make sure that `d_model` is divided into `num_heads`.) | `d_model // num_heads`                                       |
| `drop_out_rate` | `float`        | The dropout rate.                                            | `0.1`                                                        |
| `num_epochs`    | `int`          | The total number of iterations.                              | `10`                                                         |
| `beam_size`     | `int`          | The beam size. (Only used when the beam search is used at inference time.) | `8`                                                          |
| `ckpt_dir`      | `str`          | The path for saved checkpoints.                              | `saved_model`                                                |

<br/>

<hr style="background: transparent; border: 0.5px dashed;"/>

### How to run

1. Download the dataset from ["European Parliament Proceedings Parallel Corpus 1996-2011"](https://www.statmt.org/europarl/). 
	
   You can choose any parallel corpus you want. (I chose English-French for example.)
   
   Download it and extract it until you have two raw text files, `europarl-v7.SRC-TRG.SRC` and `europarl-v7.SRC-TRG.TRG`.
   
   Make `DATA_DIR` directory in the root directory and put raw texts in it.
   
   Name each ``SRC_RAW_DATA_NAME`` and ``TRG_RAW_DATA_NAME``.
   
   Of course, you can use additional datasets and just make sure that the formats/names of raw data files are same as those of above datasets. 
   
   <br/>
   
2. Install all required packages.

   ```shell
   pip install -r requirements.txt
   ```

   <br/>

3. Run `src/sentencepiece_train.py`.

   ```shell
   python src/sentencepiece_train.py
   ```

   Then there would be `SP_DIR` directory containing two sentencepiece models and two vocab files.

   Each model and vocab files are for source language and target language.

   In default setting, the structure of whole data directory should be like below.

   - `data`
     - `sp`
       - `src_sp.model`
       - `src_sp.vocab`
       - `tar_sp.model`
       - `tar_sp.vocab`
     - `src`
       - `train.txt`
       - `valid.txt`
     - `trg`
       - `train.txt`
       - `valid.txt`
     - `raw_data.src`
     - `raw_data.tar`

   <br/>

4. Run below command to train a transformer model for machine translation.

   ```shell
   python src/main.py --mode='train' --ckpt_name=CHECKPOINT_NAME
   ```

   - `--mode`: You have to specify the mode among two options, 'train' or 'inference'.
   - `--ckpt_name`: This specify the checkpoint file name. This would be the name of trained checkpoint and you can continue your training with this model in the case of resuming training. If you want to conduct training first, this parameter should be omitted. When testing, this would be the name of the checkpoint you want to test. (default: `None`)

   <br/>

5. Run below command to conduct an inference with the trained model.

   ```shell
   python src/main.py --mode='inference' --ckpt_name=CHECKPOINT_NAME --input=INPUT_TEXT --decode=DECODING_STRATEGY
   ```

   - `--input`: This is an input sequence you want to translate.
   - `--decode`: This makes the decoding algorithm into either greedy method or beam search. Make this parameter 'greedy' or 'beam'.  (default: `greedy`)

   <br/>
   
<hr style="background: transparent; border: 0.5px dashed;"/>

### References

<a id="1">[1]</a> 
*Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008)*. ([http://papers.nips.cc/paper/7181-attention-is-all-you-need](http://papers.nips.cc/paper/7181-attention-is-all-you-need))

<a id="2">[2]
*Koehn, P. (2005, September). Europarl: A parallel corpus for statistical machine translation. In MT summit (Vol. 5, pp. 79-86)*. ([http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.459.5497&rep=rep1&type=pdf](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.459.5497&rep=rep1&type=pdf))

<br/>

---
