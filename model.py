import torch
import torch.nn as nn
import torch.nn.functional as F

import random

SOS_token = 0
EOS_token = 1

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input, hidden):
        # embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.embedding(input)        
        output = embedded
        output, hidden = self.gru(output, hidden)
        
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)
    


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, MAX_LENGTH, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = MAX_LENGTH

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden[0]), 1)), dim=1)
          
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs)
        
        output = torch.cat((embedded, attn_applied.view(embedded.shape)), 1) 
               
        output = self.attn_combine(output).unsqueeze(0)   
             
        output = F.relu(output)
        
        output, hidden = self.gru(output.transpose(0, 1), hidden) 
               
        output = F.log_softmax(self.out(output.squeeze()), dim=1)
        
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class TONModel(nn.Module)  :
  def __init__(self, encoder, decoder, teacher_forcing_ratio = 0.5):
        super(TONModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder 
        self.teacher_forcing_ratio=teacher_forcing_ratio = 0.5
        
        
  def forward(self, input_tensor, target_tensor, evalute=False ):
        input_length = input_tensor.size(1)
        target_length = target_tensor.size(1)
        batch_size=input_tensor.size(0)
        
        encoder_hidden= self.encoder.initHidden(batch_size)         
           
        encoder_output, encoder_hidden = self.encoder(input_tensor, encoder_hidden)
                
        decoder_input=torch.zeros( batch_size, dtype=torch.int32).to(device)
        
        if evalute==False:
          use_teacher_forcing = False
        else:
          use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False                  

        decoder_outputs = torch.zeros((batch_size, target_length,self.decoder.output_size), device=device)
        
        
        if use_teacher_forcing: 
          for di in range(target_length):
                  
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                  decoder_input, encoder_hidden, encoder_output)
            decoder_input = target_tensor[:,di]  # Teacher forcing
            
            decoder_outputs[:,di]=decoder_output
          
        else:
          # Without teacher forcing: use its own predictions as the next input
          for di in range(target_length):            
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                  decoder_input, encoder_hidden, encoder_output)
             
            decoder_outputs[:,di]=decoder_output   
             
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            
                          
        return decoder_outputs      