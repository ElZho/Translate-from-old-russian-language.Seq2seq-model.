# Translate-from-old-russian-language.Seq2seq-model.

To start these notebooks on your computer run this comand in terminal: 

    git clone https://github.com/ElZho/Translate-from-old-russian-language.Seq2seq-model..git

To unzip zip dataset run comand in terminal: 
  
    unzip rus-eng.zip

In each notebook change path to txt files in config. 

Datasets are old slavonic russian corpus. - old_slav.txt and rus-eng.zip

I. **Baseline model** is model from pytorch totorial https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial by Sean Robertson.
There is three file:
1. preprocessing_baseline.py - function to upload, preprocess and create dataloader.
2. model_baseline.py - Encoder, Decoder and AttnDecoder - decoder with attention. Attention is on decoder inputs, not to encoder outputs.
3. Seq2Seq_translation_baseline.ipynb - main file. There is sentence length analyse, baseline train loop without validation, train loop with validation loop, prediction and attention analyses. There were used two dataset - main - old_slav and english-russian dataset.

II. **Upgrate model**:
1. model.py
2. preprocessing.py
3. Seq2Seq_translation_final.ipynb

In this upgrate model I try to realize:

- To convert data to dataloader using. This allow me to use batches to speed lerning process. More over batches can make results better. For do this I have to pad sentences. The sentences should be the same length for dataloader using. And I have to make some changes in model for load date by batches.
- To make some changes in baseline model. I join encoder and decoder with attention to single model. To do this I make encoder forward to work with whole batch, not with only one word like in baseline model. Decoder forward I keep unchanged, except that I pass for one word from whole batch, so that decoder works with tensor size equal to batch size. Padding the sentences can make model work worser, to I add mask to attention and use pad_packed_sequence in encoder to nevilate padding. I change Attention to work with encoder output, not with decoder imput like in baseline model.
- I try to use Word2vec embeddings to improve model.
- I try different loss functions and optimizers.
- I add bleu_score metric to estimate results.

III. **Upgrate model with BPE**:
1. model.py
2. preprocessing_bpe.py
3. Translation with BPE

In this upgrade I:
- create a BPE model and use bpe tokenizators for source and target datasets.

