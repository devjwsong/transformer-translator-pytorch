# transformer-nmt-pytorch
This is a machine translation project using **Transformer** introduced in *Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008)*.



I used English-French corpus provided by "European Parliament Proceedings Parallel Corpus 1996-2011", cited in publication *Koehn, P. (2005, September). Europarl: A parallel corpus for statistical machine translation. In MT summit (Vol. 5, pp. 79-86)*.

(The size of the data was so large that only 150000 samples were used for convenience.)

<br/>

---

### How to run

1. Install all required packages.
	
   ```shell
   pip install -r requirements.txt
   ```
   
   <br/>
   
2. Go to `src` directory and run `sp_process.py`.

   ```shell
   python sentencepiece_train.py
   ```

   Then in `data` directory, there would be `sp` directory containing two sentencepiece models and two vocab files.

   Each model and vocab files are for source language and target language.

   - data
     - sp
       - src_sp.model
       - src_sp.vocab
       - tar_sp.model
       - tar_sp.vocab
     - full_data.src
     - full_data.tar

   <br/>

3. Run below command to train a transformer model for machine translation.

   ```shell
   python main.py --mode='train' --ckpt_name=CHECKPOINT_NAME
   ```

   - `--mode`: You have to specify the mode among two options, 'train' or 'test'.
   - `--ckpt_name`: This specify the checkpoint file name. This would be the name of trained checkpoint and you can continue your training with this model in the case of resuming training. If you want to conduct training first, this parameter should be omitted. When testing, this would be the name of the checkpoint you want to test.

   You will get training logs and training loss as follows.

   <img src="https://user-images.githubusercontent.com/16731987/81287281-770fa280-909d-11ea-8aa2-6e4c00d36187.png" alt="Transformer in Pytorch NMT task training log."/>

   <img src="https://user-images.githubusercontent.com/16731987/81287862-77f50400-909e-11ea-86ee-f2204e0740cc.png" alt="Transformer in Pytorch NMT task training loss plots." width="70%;"/>

   <br/>

4. Run below command to test the trained model.

   ```shell
   python main.py --mode='test' --ckpt_name=CHECKPOINT_NAME --input=INPUT_TEXT --decode=DECODING_STRATEGY
   ```
   
   - `--input`: This is an input sequence you want to translate.
   - `--decode`: This makes the decoding algorithm into either greedy method or beam search. Make this parameter 'greedy' or 'beam'. 
   
   You will get the result as follows.
   
   <img src="https://user-images.githubusercontent.com/16731987/81287373-9c9cac00-909d-11ea-86a1-7024374c2b3f.png" alt="Transformer in Pytorch NMT task testing result."/>
   
   <br/>

---
