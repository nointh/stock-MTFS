import torch
from torch import nn
import torch.nn.functional as F

from common.utils import dotdict

class LSTNet(nn.Module):
    def __init__(self, args, data):
        super(LSTNet, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window;
        self.m = data.m
        self.hidR = args.hidRNN;
        self.hidC = args.hidCNN;
        self.hidS = args.hidSkip;
        self.Ck = args.CNN_kernel;
        self.skip = args.skip;
        self.pt = int((self.P - self.Ck)/self.skip)
        self.hw = args.highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m));
        self.GRU1 = nn.GRU(self.hidC, self.hidR);
        self.dropout = nn.Dropout(p = args.dropout);
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS);
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m);
        else:
            self.linear1 = nn.Linear(self.hidR, self.m);
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1);
        self.output = None;
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;
 
    def forward(self, x):
        batch_size = x.size(0);
        
        #CNN
        c = x.view(-1, 1, self.P, self.m);
        c = self.conv1(c)
        c = F.relu(c)
        # c = F.relu(self.conv1(c));
        c = self.dropout(c);
        c = torch.squeeze(c, 3);
        
        # RNN 
        r = c.permute(2, 0, 1).contiguous();
        _, r = self.GRU1(r);
        r = self.dropout(torch.squeeze(r,0));

        
        #skip-rnn
        
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous();
            s = s.view(batch_size, self.hidC, self.pt, self.skip);
            s = s.permute(2,0,3,1).contiguous();
            s = s.view(self.pt, batch_size * self.skip, self.hidC);
            _, s = self.GRUskip(s);
            s = s.view(batch_size, self.skip * self.hidS);
            s = self.dropout(s);
            r = torch.cat((r,s),1);
        
        res = self.linear1(r);
        
        #highway
        if (self.hw > 0):
            z = x[:, -self.hw:, :];
            z = z.permute(0,2,1).contiguous().view(-1, self.hw);
            z = self.highway(z);
            z = z.view(-1,self.m);
            res = res + z;
            
        if self.output is not None:
            res = self.output(res);
        return res;

def get_lstnet_multistock_model():
    args = dotdict()
    args.data = 'VN30_price.csv'
    args.model = 'LSTNet'
    args.hidCNN = 50
    args.hidRNN = 50
    args.window = 100
    args.CNN_kernel = 6
    args.highway_window = 24
    args.clip = 10.
    args.epochs = 10000
    args.batch_size = 128
    args.dropout = 0.2
    args.seed = 54321
    args.gpu = None
    args.log_interval = 2000
    args.save = 'LSTNet_multistock.pt'
    args.cuda = False
    args.optim = 'adam'
    args.lr = 0.001
    args.horizon = 1
    args.skip = 24
    args.hidSkip = 5
    args.L1Loss = False
    args.normalize = 0
    args.output_fun = None
    args.cuda = args.gpu is not None
    data = dotdict()
    data.m = 22
    model = LSTNet(args, data)
    return model
