# Translate-from-old-russian-language.Seq2seq-model.

Dataset is old slavonic russian corpus. - old_slav.txt

I. Baseline model is model from pytorch totorial https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial by Sean Robertson.
There is three file:
1. preprocessing_baseline.py - function to upload, preprocess and create dataloader.
2. model_baseline.py - Encoder, Decoder and AttnDecoder - decoder with attention. Attention is on decoder inputs, not to encoder outputs.
3. Seq2Seq_translation_baseline - main file. There is sentence length analyse, baseline train loop without validation, train loop with validation loop, prediction and attention analyses. There were used two dataset - main - old_slav and english-russian dataset.

Main - NLP_translate-1.
Function to extract the data, tokinize sentences - preprocessing.py
Model - model.py
