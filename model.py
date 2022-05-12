import torch
import torch.nn as nn
import torch.nn.functional as F

import random

SOS_token = 0
EOS_token = 1

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding=None):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        if embedding== None:
          self.embedding = nn.Embedding(input_size, hidden_size)
        else:
          self.embedding = embedding
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input, input_len, hidden):
        # embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.embedding(input) #batch_size x max_length x embedding_size
        tot_len=embedded.shape[1]
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_len.to('cpu'), batch_first=True, enforce_sorted=False)       
        output = embedded        
        output, hidden = self.gru(output, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=tot_len) #batch_size x length_of_sentence x embedding_size
        
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)
    


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, MAX_LENGTH=340, dropout_p=0.1, embedding=None):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = MAX_LENGTH
        if embedding== None:
          self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        else:
          self.embedding = embedding  
        #self.attn = nn.Linear(self.hidden_size * 2, self.max_length)# original
        self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.v=nn.Linear(self.hidden_size, 1, bias=False)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, mask):
        # input: batch_size x 1 (one word) or batch_size
        
        embedded = self.embedding(input).view(input.shape[0], 1, -1) #batch_size x output_size x 1
        embedded = self.dropout(embedded)
        
        ##############################
        # attention from original model

        # attention=self.attn(torch.cat((embedded[:,0], hidden[0]), 1)) #batch_size x max_len
        # attention=attention.masked_fill(attention==0, -1e10)
        # attn_weights = F.softmax(attention, dim=1) #batch_size x max_len


        # print("attn_weights", attn_weights.shape)  
        # print('attn_weights.unsqueeze(1)', attn_weights.unsqueeze(1).shape, 'encoder_outputs',
        #                          encoder_outputs.shape)

        ###############################
        #attention remade
        
        attention=self.attn(torch.cat((encoder_outputs, hidden.repeat(self.max_length, 1, 1).permute(1, 0, 2)), 2)) #batch_size x max_len x hidden_size
        attention=self.v(attention).squeeze(2) #batch_size x max_len        
        attention=attention.masked_fill(mask==0, -1e10)
        attn_weights = F.softmax(attention, dim=1) #batch_size x max_len
        ###############################

        
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs)  #batch x 1 (one word) x embedding_size
        
        output = torch.cat((embedded.view(embedded.shape[0],-1), attn_applied.view(embedded.shape[0],-1)), 1) # batch_size x 2*emb_size    
        output = self.attn_combine(output).unsqueeze(0)   #1 (one word) x batch_size x embedding_size      
        output = F.relu(output)
        
        output, hidden = self.gru(output.transpose(0, 1), hidden) #batch x 1 (one word) x embedding_size       
        output = F.log_softmax(self.out(output[:,0,:]), dim=1) #batch x vocabalary_size
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class TONModel(nn.Module)  :
  def __init__(self, encoder, decoder, teacher_forcing_ratio = 0.5):
        super(TONModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder 
        self.teacher_forcing_ratio=teacher_forcing_ratio = 0.5
  def create_mask(self, input_tensor):
        mask = (input_tensor != 0).permute(0, 1)
        return mask     
        
  def forward(self, input_tensor, input_len, target_tensor, evalute=False ):
        
        input_length = input_tensor.size(1)
        target_length = target_tensor.size(1)
        batch_size=input_tensor.size(0)

        input_mask=self.create_mask(input_tensor)
        
        encoder_hidden= self.encoder.initHidden(batch_size)         
           
        encoder_output, encoder_hidden = self.encoder(input_tensor, input_len, encoder_hidden)
                
        decoder_input=torch.zeros( batch_size, dtype=torch.int32).to(device)
        
        if evalute==False:
          use_teacher_forcing = False
        else:
          use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False                  

        decoder_outputs = torch.zeros((batch_size, target_length,self.decoder.output_size), device=device)
        
        for di in range(target_length):
                  
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                  decoder_input, encoder_hidden, encoder_output, input_mask)                        
            decoder_outputs[:,di]=decoder_output
        
            if use_teacher_forcing: 
              decoder_input = target_tensor[:,di]  # Teacher forcing
              
            else:
              # Without teacher forcing: use its own predictions as the next input
               
              topv, topi = decoder_output.topk(1,1)
              decoder_input = topi.detach() # detach from history as input
            if  decoder_input.sum() ==0:
               break
                
        return decoder_outputs,   decoder_attention    
