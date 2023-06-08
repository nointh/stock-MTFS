import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
import pandas as pd
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



class LSTM(nn.Module):
    def __init__(
            self,
            input_size,
            lstm_hid_size,
            linear_hid_size,
            output_horizon=1,
            n_layers=1,
        ):
        super().__init__()
        self.n_layers = n_layers
        self.lstm_hid_size = lstm_hid_size
        self.output_horizon = output_horizon

        self.lstm = nn.LSTM(input_size, lstm_hid_size, n_layers, \
                            bias=True, batch_first=True) 

        self.linear = nn.Sequential(
            nn.Linear(lstm_hid_size, linear_hid_size),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(linear_hid_size, output_horizon) 
        )
        

    def forward(self, x):
        batch_size, obs_len, f_dim = x.size()

        ht = torch.zeros(self.n_layers, batch_size, self.lstm_hid_size).to(device)
        ct = ht.clone()
        for t in range(obs_len):
            xt = x[:, t, :].view(batch_size, 1, -1)
            out, (ht, ct) = self.lstm(xt, (ht, ct))
            htt = ht.permute(1, 0, 2)
            htt = htt[:, -1, :]

        htt = htt.reshape(batch_size, -1)
        out = self.linear(htt).unsqueeze(-1)

        return out

class LSTNet(nn.Module):
    def __init__(self, 
                 window, 
                 n_features, 
                 hidden_RNN, 
                 hidden_CNN,
                 hidden_skip,
                 CNN_kernel,
                 skip,
                 highway_window,
                 dropout,
                 output_func):
        super(LSTNet, self).__init__()
        self.window = window
        self.n_features = n_features
        self.hidden_RNN = hidden_RNN
        self.hidden_CNN = hidden_CNN
        self.hidden_skip = hidden_skip
        self.CNN_kernel = CNN_kernel
        self.skip = skip
        self.pt = int((self.window - self.CNN_kernel)/self.skip)
        self.highway_window = highway_window
        self.dropout = dropout
        
        self.conv1 = nn.Conv2d(1, self.hidden_CNN, (self.CNN_kernel, self.n_features))
        self.GRU1 = nn.GRU(self.hidden_CNN, self.hidden_RNN)
        self.dropout = nn.Dropout(p = self.dropout)
        if self.skip > 0:
            self.GRU_skip = nn.GRU(self.hidden_CNN, self.hidden_skip)
            self.linear_1 = nn.Linear(self.hidden_RNN + self.skip * self.hidden_skip, self.n_features)
        else:
            self.linear_1 = nn.Linear(self.hidden_RNN, self.n_features)
        if self.highway_window > 0:
            self.highway = nn.Linear(self.highway_window, 1)
        
        self.output = None
        if output_func == 'sigmoid':
            self.output = nn.functional.sigmoid;
        if output_func == 'tanh':
            self.output = nn.functional.tanh
    def forward(self, x):
        batch_size = x.size(0)
        
        #CNN
        c = x.view(-1, 1, self.window, self.n_features)
        c = self.conv1(c)
        c = nn.functional.relu(c)
        # c = F.relu(self.conv1(c));
        c = self.dropout(c)
        c = torch.squeeze(c, 3)
        
        # RNN 
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r,0))

        
        #skip-rnn
        
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous();
            s = s.view(batch_size, self.hidden_CNN, self.pt, self.skip)
            s = s.permute(2,0,3,1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidden_CNN)
            _, s = self.GRU_skip(s)
            s = s.view(batch_size, self.skip * self.hidden_skip)
            s = self.dropout(s)
            r = torch.cat((r,s),1)
        
        res = self.linear_1(r)
        
        #highway
        if self.highway_window > 0:
            z = x[:, -self.highway_window:, :];
            z = z.permute(0,2,1).contiguous().view(-1, self.highway_window)
            z = self.highway(z)
            z = z.view(-1,self.n_features)
            res = res + z
            
        if self.output:
            res = self.output(res)
        return res


hidden_size = 128
num_layers = 1
learning_rate = 0.001
epochs = 10
input_sequence = 10
output_sequence = 1
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

model = LSTM(n_features, hidden_size, 128, output_sequence, num_layers)
# model = LSTNet(window=168, 
#                n_features=n_features, 
#                hidden_RNN=50, 
#                hidden_CNN=50, 
#                hidden_skip=5,
#                CNN_kernel=6,
#                skip=24, 
#                highway_window=24, dropout=0.2, output_func=None)
criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), learning_rate)
optimizer = Optim(model.parameters(), 'adam', learning_rate, 10)
model.train()
for epoch in range(epochs):
    for i, (input, true_output) in enumerate(DataLoader(train_dataset, batch_size=batch_size)):
        model.zero_grad()
        output = model(input)
        # true_output = torch.squeeze(true_output[:, :, [0]], 1)
        true_output = true_output[:, :, [1]]
        loss = criterion(output, true_output)
        # optimizer.zero_grad()
        loss.backward()
        print('epoch number {}, iter {}, LOSS {}'.format(epoch+1, i+1, loss.item()) )
        optimizer.step()
    if (i+1) % 10 == 0:
        print (f'Epoch [{epoch+1}], Loss: {loss.item():.4f}')




print("Result ================================")
for i, (input, true_output) in enumerate(DataLoader(test_dataset, batch_size=300)):
    model.eval()
    model.zero_grad()
    output = model(input)
    true_output = torch.squeeze(true_output, 1)
    
    pred_price = output.squeeze().detach().numpy()
    true_price = true_output[:, -1].detach().numpy()
    print(pred_price)
    print(true_price)
    import numpy as np
    print(np.sqrt(np.mean((pred_price-true_price)**2)))
    