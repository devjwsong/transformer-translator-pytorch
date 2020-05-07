# transformer-pytorch
This is a machine translation project using **Transformer** introduced in *Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008)*.



I used English-French corpus provided by "European Parliament Proceedings Parallel Corpus 1996-2011", cited in publication *Koehn, P. (2005, September). Europarl: A parallel corpus for statistical machine translation. In MT summit (Vol. 5, pp. 79-86)*.

(The size of the data was so large that only 50000 samples were used for convenience.)

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
   python main.py --mode='train'
   ```

   <br/>

   You will get training logs and training loss as follows.

   <img src="https://drive.google.com/uc?id=1opDUPfTj23dPeirAsmoeL71fLK4jzZlk" style="margin-left: 0;"/>

   <img src="https://drive.google.com/uc?id=1PEhEkeecfHAjXx1H7J5JsFqvfbI1IWLB" style="width: 70%; margin-left: 0;">

   <br/>

4. Run below command to test the trained model.

   ```shell
   python main.py --mode='test' --model_name=MODEL_FILE_NAME --input=INPUT_TEXT
   ```

   <br/>
   
   You will get the result as follows.
   
   <img src="https://drive.google.com/uc?id=1D_dIuORGxmz1LQuuzMlFur8DMOQWuHU6" style="margin-left: 0;"/>
   
   <br/>

---