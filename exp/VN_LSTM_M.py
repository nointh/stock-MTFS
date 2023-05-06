import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas as pd

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
        self.data = torch.tensor(df.values).float()
    
    def __len__(self):
        return len(self.data) - self.input_sequence - self.output_sequence + 1
    
    def __getitem__(self, index):
        x_begin = index
        x_end = x_begin + self.input_sequence
        y_begin = x_end
        y_end = y_begin + self.output_sequence
        return self.data[x_begin: x_end], self.data[y_begin:y_end]



class LSTM(nn.Module):
    def __init__(self, window, hidden_size, num_layers, n_features):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.window = window
        self.n_features = n_features
        self.lstm = nn.LSTM(
            input_size=n_features, 
            hidden_size=hidden_size, 
            batch_first=True, 
            num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, X):
        batch_size = X.size(0)
        h_state = self.init_state(X)
        c_state = self.init_state(X)
        output, (hn, _) = self.lstm(X, (h_state, c_state))
        out = self.fc(hn)
        out = out.permute(1, 0, 2)
        return out
    
    def init_state(self, X):
        return torch.zeros(self.num_layers, X.size(0), self.hidden_size)

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
epochs = 100
input_sequence = 168
output_sequence = 1
batch_size=128
train_dataset = StockPriceDataset('exp/data/VN30.csv', input_sequence, output_sequence)
val_dataset = StockPriceDataset('exp/data/VN30.csv', input_sequence, output_sequence, flag='val')
test_dataset = StockPriceDataset('exp/data/VN30.csv', input_sequence, output_sequence, flag='test')
n_features = train_dataset.n_features

# model = LSTM(input_sequence, hidden_size, num_layers, n_features)
model = LSTNet(window=168, 
               n_features=n_features, 
               hidden_RNN=50, 
               hidden_CNN=50, 
               hidden_skip=5,
               CNN_kernel=6,
               skip=24, 
               highway_window=24, dropout=0.2, output_func=None)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), learning_rate)
for epoch in range(epochs):
    model.train()
    for i, (input, true_output) in enumerate(DataLoader(train_dataset, batch_size=batch_size)):
        output = model(input)
        true_output = torch.squeeze(true_output, 1)
        loss = criterion(output, true_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            print (f'Epoch [{epoch+1}], Loss: {loss.item():.4f}')
