import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
device = 'cpu'
class StockPriceDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 input_sequence: int,
                 output_sequence: int,
                 flag: str='train',
                 train_split: float = 0.7,
                 val_split: float = 0.1
                 ):
        """
        :param data_path: file path of the csv file
        :param input_sequence: window size for input
        :param output_sequence: horizon size for output
        :param flag: configure for data purpose like train or validation or test
        """
        self.input_sequence = input_sequence
        self.output_sequence = output_sequence
        df = pd.read_csv(data_path, index_col=0)
        df.reset_index(drop=True, inplace=True)
        self.n_features = df.shape[1]
        num_train = int(len(df)*train_split)
        num_valid = int(len(df)*val_split)
        if flag == 'val':
            df = df[num_valid:num_train+num_valid]
        elif flag == 'test':
            df = df[num_train+num_valid:]
        else:
            df = df[:num_train]
        self.scaler = StandardScaler()
        self.scaler.fit_transform(df[:num_train].values)
        self.data = torch.tensor(self.scaler.transform(df.values)).float()
    
    def __len__(self):
        return len(self.data) - self.input_sequence - self.output_sequence + 1
    
    def __getitem__(self, index):
        x_begin = index
        x_end = x_begin + self.input_sequence
        y_begin = x_end
        y_end = y_begin + self.output_sequence
        return self.data[x_begin: x_end], self.data[y_begin:y_end]



class lstm_encoder(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, input_size, hidden_size, num_layers = 1):
        
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # define LSTM layer
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers)

    def forward(self, x_input):
        
        '''
        : param x_input:               input of shape (seq_len, # in batch, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence 
        '''
        
        lstm_out, self.hidden = self.lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))
        
        return lstm_out, self.hidden     
    
    def init_hidden(self, batch_size):
        
        '''
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state 
        '''
        
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


class lstm_decoder(nn.Module):
    ''' Decodes hidden state output by encoder '''
    
    def __init__(self, input_size, hidden_size, num_layers = 1):

        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        
        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers)
        self.linear = nn.Linear(hidden_size, input_size)           

    def forward(self, x_input, encoder_hidden_states):
        
        '''        
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence 
 
        '''
        
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(0))     
        
        return output, self.hidden

class lstm_seq2seq(nn.Module):
    ''' train LSTM encoder-decoder and make predictions '''
    
    def __init__(self, input_size, hidden_size):

        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        '''

        super(lstm_seq2seq, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = lstm_encoder(input_size = input_size, hidden_size = hidden_size)
        self.decoder = lstm_decoder(input_size = input_size, hidden_size = hidden_size)


    def train_model(self, input_tensor, target_tensor, n_epochs, target_len, batch_size, training_prediction = 'recursive', teacher_forcing_ratio = 0.5, learning_rate = 0.01, dynamic_tf = False):
        
        '''
        train lstm encoder-decoder
        
        : param input_tensor:              input data with shape (seq_len, # in batch, number features); PyTorch tensor    
        : param target_tensor:             target data with shape (seq_len, # in batch, number features); PyTorch tensor
        : param n_epochs:                  number of epochs 
        : param target_len:                number of values to predict 
        : param batch_size:                number of samples per gradient update
        : param training_prediction:       type of prediction to make during training ('recursive', 'teacher_forcing', or
        :                                  'mixed_teacher_forcing'); default is 'recursive'
        : param teacher_forcing_ratio:     float [0, 1) indicating how much teacher forcing to use when
        :                                  training_prediction = 'teacher_forcing.' For each batch in training, we generate a random
        :                                  number. If the random number is less than teacher_forcing_ratio, we use teacher forcing.
        :                                  Otherwise, we predict recursively. If teacher_forcing_ratio = 1, we train only using
        :                                  teacher forcing.
        : param learning_rate:             float >= 0; learning rate
        : param dynamic_tf:                use dynamic teacher forcing (True/False); dynamic teacher forcing
        :                                  reduces the amount of teacher forcing for each epoch
        : return losses:                   array of loss function for each epoch
        '''
        
        # initialize array of losses 
        losses = np.full(n_epochs, np.nan)

        optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        criterion = nn.MSELoss()

        # calculate number of batch iterations
        n_batches = int(input_tensor.shape[1] / batch_size)

        with trange(n_epochs) as tr:
            for it in tr:
                
                batch_loss = 0.
                batch_loss_tf = 0.
                batch_loss_no_tf = 0.
                num_tf = 0
                num_no_tf = 0

                for b in range(n_batches):
                    # select data 
                    input_batch = input_tensor[:, b: b + batch_size, :]
                    target_batch = target_tensor[:, b: b + batch_size, :]

                    # outputs tensor
                    outputs = torch.zeros(target_len, batch_size, input_batch.shape[2])

                    # initialize hidden state
                    encoder_hidden = self.encoder.init_hidden(batch_size)

                    # zero the gradient
                    optimizer.zero_grad()

                    # encoder outputs
                    encoder_output, encoder_hidden = self.encoder(input_batch)

                    # decoder with teacher forcing
                    decoder_input = input_batch[-1, :, :]   # shape: (batch_size, input_size)
                    decoder_hidden = encoder_hidden

                    if training_prediction == 'recursive':
                        # predict recursively
                        for t in range(target_len): 
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[t] = decoder_output
                            decoder_input = decoder_output

                    if training_prediction == 'teacher_forcing':
                        # use teacher forcing
                        if random.random() < teacher_forcing_ratio:
                            for t in range(target_len): 
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[t] = decoder_output
                                decoder_input = target_batch[t, :, :]

                        # predict recursively 
                        else:
                            for t in range(target_len): 
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[t] = decoder_output
                                decoder_input = decoder_output

                    if training_prediction == 'mixed_teacher_forcing':
                        # predict using mixed teacher forcing
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[t] = decoder_output
                            
                            # predict with teacher forcing
                            if random.random() < teacher_forcing_ratio:
                                decoder_input = target_batch[t, :, :]
                            
                            # predict recursively 
                            else:
                                decoder_input = decoder_output

                    # compute the loss 
                    loss = criterion(outputs, target_batch)
                    batch_loss += loss.item()
                    
                    # backpropagation
                    loss.backward()
                    optimizer.step()

                # loss for epoch 
                batch_loss /= n_batches 
                losses[it] = batch_loss

                # dynamic teacher forcing
                if dynamic_tf and teacher_forcing_ratio > 0:
                    teacher_forcing_ratio = teacher_forcing_ratio - 0.02 

                # progress bar 
                tr.set_postfix(loss="{0:.3f}".format(batch_loss))
                    
        return losses

    def predict(self, input_tensor, target_len):
        
        '''
        : param input_tensor:      input data (seq_len, input_size); PyTorch tensor 
        : param target_len:        number of target values to predict 
        : return np_outputs:       np.array containing predicted values; prediction done recursively 
        '''

        # encode input_tensor
        input_tensor = input_tensor.unsqueeze(1)     # add in batch size of 1
        encoder_output, encoder_hidden = self.encoder(input_tensor)

        # initialize tensor for predictions
        outputs = torch.zeros(target_len, input_tensor.shape[2])

        # decode input_tensor
        decoder_input = input_tensor[-1, :, :]
        decoder_hidden = encoder_hidden
        
        for t in range(target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output.squeeze(0)
            decoder_input = decoder_output
            
        np_outputs = outputs.detach().numpy()
        
        return np_outputs


hidden_size = 256
num_layers = 1
learning_rate = 0.001
epochs = 100
input_sequence = 48
output_sequence = 48
batch_size=128
train_dataset = StockPriceDataset('exp/data/VN30_price.csv', input_sequence, output_sequence)
val_dataset = StockPriceDataset('exp/data/VN30_price.csv', input_sequence, output_sequence, flag='val')
test_dataset = StockPriceDataset('exp/data/VN30_price.csv', input_sequence, output_sequence, flag='test')
n_features = train_dataset.n_features


class Optim(object):

    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, method, lr, max_grad_norm, lr_decay=1, start_decay_at=None):
        self.params = list(params)  # careful: params may be a generator
        self.last_ppl = None
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False

        self._makeOptimizer()

    def step(self):
        # Compute gradients norm.
        grad_norm = 0
        for param in self.params:
            grad_norm += math.pow(param.grad.data.norm(), 2)

        grad_norm = math.sqrt(grad_norm)
        if grad_norm > 0:
            shrinkage = self.max_grad_norm / grad_norm
        else:
            shrinkage = 1.

        for param in self.params:
            if shrinkage < 1:
                param.grad.data.mul_(shrinkage)

        self.optimizer.step()
        return grad_norm

    # decay learning rate if val perf does not improve or we hit the start_decay_at limit
    def updateLearningRate(self, ppl, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)
        #only decay for one epoch
        self.start_decay = False

        self.last_ppl = ppl

        self._makeOptimizer()
        
model = DSTP_RNN(n_features-1, hidden_size, hidden_size, input_sequence, output_sequence)
criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), learning_rate)
optimizer = Optim(model.parameters(), 'adam', learning_rate, 10)
model.train()
for epoch in range(epochs):
    test_loss = 0
    n_item = 0
    for i, (input, true_output) in enumerate(DataLoader(train_dataset, batch_size=batch_size)):
        model.zero_grad()
        x = input[:, :, :-1]
        y_history = input[:, :, -1:]
        output = model(x, y_history)
        # true_output = torch.squeeze(true_output[:, :, [0]], 1)
        true_output = true_output[:, :, -1]
        loss = criterion(output, true_output)
        test_loss += loss.item()
        n_item += 1
        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print('epoch number {}, iter {}, LOSS {}'.format(epoch+1, i+1, test_loss/n_item) )
    if (epoch+1) % 10 == 0:
        print (f'Epoch [{epoch+1}], Loss: {loss.item():.4f}')
        pred = []
        true = []
        for i, (input, true_output) in enumerate(DataLoader(val_dataset, batch_size=batch_size)):
            model.eval()
            model.zero_grad()
            x = input[:, :, :-1]
            y_history = input[:, :, -1:]
            output = model(x, y_history)
            # true_output = torch.squeeze(true_output, 1)
            pred_price = output.detach().numpy()
            # pred_price = output.squeeze().detach().numpy()
            true_price = true_output[:, :, -1].detach().numpy()
            pred.extend(pred_price)
            true.extend(true_price)
        print('Val RMSE')
        print(np.sqrt(np.mean((np.array(pred)-np.array(true))**2)))





print("Result ================================")
pred = []
true = []
for i, (input, true_output) in enumerate(DataLoader(test_dataset, batch_size=300)):
    model.eval()
    model.zero_grad()
    x = input[:, :, :-1]
    y_history = input[:, :, -1:]
    output = model(x, y_history)
    # true_output = torch.squeeze(true_output, 1)
    pred_price = output.detach().tolist()
    pred.extend(pred_price)
    # pred_price = output.squeeze().detach().numpy()
    true_price = true_output[:, :, -1].detach().tolist()
    true.extend(true_price)

# print(pred_price)
# print(true_price)
import matplotlib.pyplot as plt
plt.plot(np.array(true))
plt.plot(np.array(pred))
plt.save()
print('RMSE')
print(np.sqrt(np.mean((np.array(pred_price)-np.array(true_price))**2)))
    
# %%
