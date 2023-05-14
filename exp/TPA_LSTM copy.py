"""
   Contains all the utility functions that would be needed
   1. _normalized
   2. _split
   3._batchify
   4. get_batches
   """
import argparse
import math
import time
import importlib

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, cuda, horizon, window, normalize=2):
        self.cuda = cuda;
        self.window_length = window;
        self.horizon = horizon
        # fin = open(file_name);
        # self.original_data = np.loadtxt(fin, delimiter=',');
        self.original_data = pd.read_csv(file_name, index_col=0).to_numpy()
        self.normalized_data = np.zeros(self.original_data.shape);
        self.original_rows, self.original_columns = self.normalized_data.shape;
        self.normalize = 2
        self.scale = np.ones(self.original_columns);
        self._normalized(normalize);

        #after this step train, valid and test have the respective data, split from original_data according to the ratios
        self._split(int(train * self.original_rows), int((train + valid) * self.original_rows), self.original_rows);

        self.scale = torch.from_numpy(self.scale).float();
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.original_columns);

        if self.cuda:
            self.scale = self.scale.cuda();
        self.scale = Variable(self.scale);

        #rse and rae must be some sort of errors for now, will come back to them later
        self.rse = normal_std(tmp);
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)));

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.normalized_data = self.original_data

        if (normalize == 1):
            self.normalized_data = self.original_data / np.max(self.original_data);

        # normalized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.original_columns):
                self.scale[i] = np.max(np.abs(self.original_data[:, i]));
                self.normalized_data[:, i] = self.original_data[:, i] / np.max(np.abs(self.original_data[:, i]));

    def _split(self, train, valid, test):

        train_set = range(self.window_length + self.horizon - 1, train);
        valid_set = range(train, valid);
        test_set = range(valid, self.original_rows);
        self.train = self._batchify(train_set, self.horizon);
        self.valid = self._batchify(valid_set, self.horizon);
        self.test = self._batchify(test_set, self.horizon);

    def _batchify(self, idx_set, horizon):

        n = len(idx_set);
        X = torch.zeros((n, self.window_length, self.original_columns));
        Y = torch.zeros((n, self.original_columns));

        for i in range(n):
            end = idx_set[i] - self.horizon + 1;
            start = end - self.window_length;
            X[i, :, :] = torch.from_numpy(self.normalized_data[start:end, :]);
            Y[i, :] = torch.from_numpy(self.normalized_data[idx_set[i], :]);

        """
            Here matrix X is 3d matrix where each of it's 2d matrix is the separate window which has to be sent in for training.
            Y is validation.           
        """
        return [X, Y];

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt];
            Y = targets[excerpt];
            if (self.cuda):
                X = X.cuda();
                Y = Y.cuda();
            yield Variable(X), Variable(Y);
            start_idx += batch_size


class TPA_LSTM_Modified(nn.Module):
    def __init__(self, args, data):
        super(TPA_LSTM_Modified, self).__init__()
        self.use_cuda = args.cuda
        self.window_length = args.window;  # window, read about it after understanding the flow of the code...What is window size? --- temporal window size (default 24 hours * 7)
        self.original_columns = data.original_columns  # the number of columns or features
        self.hidR = args.hidRNN;
        self.hidden_state_features = args.hidden_state_features
        self.hidC = args.hidCNN;
        self.hidS = args.hidSkip;
        self.Ck = args.CNN_kernel;  # the kernel size of the CNN layers
        self.skip = args.skip;
        self.pt = (self.window_length - self.Ck) // self.skip
        self.hw = args.highway_window
        self.num_layers_lstm = args.num_layers_lstm
        self.hidden_state_features_uni_lstm = args.hidden_state_features_uni_lstm
        self.attention_size_uni_lstm = args.attention_size_uni_lstm
        self.num_layers_uni_lstm = args.num_layers_uni_lstm
        self.lstm = nn.LSTM(input_size=self.original_columns, hidden_size=self.hidden_state_features,
                            num_layers=self.num_layers_lstm,
                            bidirectional=False);
        self.uni_lstm = nn.LSTM(input_size=1, hidden_size=args.hidden_state_features_uni_lstm,
                            num_layers=args.num_layers_uni_lstm,
                            bidirectional=False);
        self.compute_convolution = nn.Conv2d(1, self.hidC, kernel_size=(
            self.Ck, self.hidden_state_features))  # hidC are the num of filters, default value of Ck is one
        self.attention_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.hidC, self.hidden_state_features, requires_grad=True)) #, device='cuda'
        self.context_vector_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.hidden_state_features, self.hidC, requires_grad=True)) #, device='cuda'
        self.final_state_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.hidden_state_features, self.hidden_state_features, requires_grad=True)) #, device='cuda'
        self.final_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.original_columns, self.hidden_state_features, requires_grad=True)) #, device='cuda'

        self.attention_matrix_uni_lstm = nn.Parameter(
            torch.ones(args.batch_size, self.hidden_state_features_uni_lstm, self.hidden_state_features_uni_lstm, self.original_columns, requires_grad=True))
        self.context_vector_matrix_uni_lstm = nn.Parameter(
            torch.ones(args.batch_size, self.hidden_state_features_uni_lstm, self.hidden_state_features_uni_lstm, self.original_columns,
                       requires_grad=True))
        self.final_hidden_uni_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.hidden_state_features_uni_lstm, self.hidden_state_features_uni_lstm,
                       self.original_columns,
                       requires_grad=True))
        self.final_uni_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.hidden_state_features, self.hidden_state_features_uni_lstm,
                       self.original_columns,
                       requires_grad=True))


        self.bridge_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.hidden_state_features, self.hidden_state_features,
                       requires_grad=True))


        torch.nn.init.xavier_uniform(self.attention_matrix)
        torch.nn.init.xavier_uniform(self.context_vector_matrix)
        torch.nn.init.xavier_uniform(self.final_state_matrix)
        torch.nn.init.xavier_uniform(self.final_matrix)
        torch.nn.init.xavier_uniform(self.attention_matrix_uni_lstm)
        torch.nn.init.xavier_uniform(self.context_vector_matrix_uni_lstm)
        torch.nn.init.xavier_uniform(self.final_hidden_uni_matrix)
        torch.nn.init.xavier_uniform(self.final_state_matrix)
        torch.nn.init.xavier_uniform(self.bridge_matrix)


        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.original_columns));  # kernel size is size for the filters
        self.GRU1 = nn.GRU(self.hidC, self.hidR);
        self.dropout = nn.Dropout(p=args.dropout);
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS);
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.original_columns);
        else:
            self.linear1 = nn.Linear(self.hidR, self.original_columns);
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1);
        self.output = None;
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;

    def forward(self, x):
        batch_size = x.size(0);
        x = x

        """
           Step 1. First step is to feed this information to LSTM and find out the hidden states

            General info about LSTM:

            Inputs: input, (h_0, c_0)
        - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
          of the input sequence.
          The input can also be a packed variable length sequence.
          See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
          :func:`torch.nn.utils.rnn.pack_sequence` for details.
        - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial hidden state for each element in the batch.
          If the RNN is bidirectional, num_directions should be 2, else it should be 1.
        - **c_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial cell state for each element in the batch.

          If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.


    Outputs: output, (h_n, c_n)
        - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
          containing the output features `(h_t)` from the last layer of the LSTM,
          for each t. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
          given as the input, the output will also be a packed sequence.

          For the unpacked case, the directions can be separated
          using ``output.view(seq_len, batch, num_directions, hidden_size)``,
          with forward and backward being direction `0` and `1` respectively.
          Similarly, the directions can be separated in the packed case.
        - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the hidden state for `t = seq_len`.

          Like *output*, the layers can be separated using
          ``h_n.view(num_layers, num_directions, batch, hidden_size)`` and similarly for *c_n*.
        - **c_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the cell state for `t = seq_len`

        """
        ##Incase in future bidirectional lstms are to be used, size of hn would needed to be modified a little (as output is of size (num_layers * num_directions, batch, hidden_size))
        input_to_lstm = x.permute(1, 0, 2).contiguous()  # input to lstm is of shape (seq_len, batch, features) (x is of shape (batch_size, seq_length, features))
        lstm_hidden_states, (h_all, c_all) = self.lstm(input_to_lstm)
        hn = h_all[-1].view(1, h_all.size(1), h_all.size(2))

        """
            Step 2. Apply convolution on these hidden states. As in the paper TPA-LSTM, these filters are applied on the rows of the hidden state
        """
        output_realigned = lstm_hidden_states.permute(1, 0, 2).contiguous()
        hn = hn.permute(1, 0, 2).contiguous()
        # cn = cn.permute(1, 0, 2).contiguous()
        input_to_convolution_layer = output_realigned.view(-1, 1, self.window_length, self.hidden_state_features);
        convolution_output = F.relu(self.compute_convolution(input_to_convolution_layer));
        convolution_output = self.dropout(convolution_output);


        """
            Step 3. Apply attention on this convolution_output
        """
        convolution_output = convolution_output.squeeze(3)

        """
                In the next 10 lines, padding is done to make all the batch sizes as the same so that they do not pose any problem while matrix multiplication
                padding is necessary to make all batches of equal size
        """
        final_hn = torch.zeros(self.attention_matrix.size(0), 1, self.hidden_state_features)
        input = torch.zeros(self.attention_matrix.size(0), x.size(1), x.size(2))
        final_convolution_output = torch.zeros(self.attention_matrix.size(0), self.hidC, self.window_length)
        diff = 0
        if (hn.size(0) < self.attention_matrix.size(0)):
            final_hn[:hn.size(0), :, :] = hn
            final_convolution_output[:convolution_output.size(0), :, :] = convolution_output
            input[:x.size(0), :, :] = x
            diff = self.attention_matrix.size(0) - hn.size(0)
        else:
            final_hn = hn
            final_convolution_output = convolution_output
            input = x

        """
           final_hn, final_convolution_output are the matrices to be used from here on
        """
        convolution_output_for_scoring = final_convolution_output.permute(0, 2, 1).contiguous()
        final_hn_realigned = final_hn.permute(0, 2, 1).contiguous()
        convolution_output_for_scoring = convolution_output_for_scoring
        final_hn_realigned = final_hn_realigned
        mat1 = torch.bmm(convolution_output_for_scoring, self.attention_matrix)
        scoring_function = torch.bmm(mat1, final_hn_realigned)
        alpha = torch.nn.functional.sigmoid(scoring_function)
        context_vector = alpha * convolution_output_for_scoring
        context_vector = torch.sum(context_vector, dim=1)

        """
           Step 4. Compute the output based upon final_hn_realigned, context_vector
        """
        context_vector = context_vector.view(-1, self.hidC, 1)
        h_intermediate = torch.bmm(self.final_state_matrix, final_hn_realigned) + torch.bmm(self.context_vector_matrix, context_vector)

        """
            Up until now TPA-LSTM has been implemented in pytorch
            
            Modification
            Background: TPA-LSTM guys use all features together and stack them up during the RNN stage. This treats one time step as one entity, killing the individual
            properties of each variable. At the very end, the variable we are predicting for must depend on itself the most.
            
            Proposition: Use lstm on each variable independently (assuming them to be independent). Using the hidden state of each time step, along with the hidden state
            of the CNN, now apply the same attention model. This method will preserve the identiy of the individual series by not considering all of them as one state.   
        """
        individual_all_hidden_states = None
        individual_last_hidden_state = None
        for feature_num in range(0, self.original_columns):
            individual_feature = input[:, :, feature_num].view(input.size(0), input.size(1), -1).permute(1, 0, 2).contiguous()
            uni_output, (uni_hn, uni_cn) = self.uni_lstm(individual_feature)  #Output of hn is of the size  (num_layers * num_directions, batch, hidden_size)  |  num_layers = 2 in bidirectional lstm
            if(feature_num == 0):
                individual_all_hidden_states = uni_output.permute(1, 0, 2).contiguous()
                individual_last_hidden_state = uni_hn[-1].view(1, uni_hn.size(1), uni_hn.size(2)).permute(1, 0, 2).contiguous()
            else:
                individual_all_hidden_states = torch.cat((individual_all_hidden_states, uni_output.permute(1, 0, 2).contiguous()), 1)
                individual_last_hidden_state = torch.cat((individual_last_hidden_state, uni_hn[-1].view(1, uni_hn.size(1), uni_hn.size(2)).permute(1, 0, 2).contiguous()), 1)

          ## *****************DIMENSIONS OF individual_all_hidden_states are (batch_size, time series length/window size, hidden_state_features, total univariate series)*****************##
         ## *****************DIMENSIONS OF individual_last_hidden_state are (batch_size, 1, hidden_state_features, total univariate series)*****************##

        individual_all_hidden_states = individual_all_hidden_states.view(input.size(0), input.size(1), self.hidden_state_features_uni_lstm, -1)
        individual_last_hidden_state = individual_last_hidden_state.view(input.size(0), 1, self.hidden_state_features_uni_lstm, -1)
        """
        Calculate the attention score for all of these
        """
        univariate_attended = []
        h_output=None
        for feature_num in range(0, self.original_columns):
            attention_matrix_uni = self.attention_matrix_uni_lstm[:, :, :, feature_num]
            context_vector_matrix_uni = self.context_vector_matrix_uni_lstm[:, :, :, feature_num]
            hidden_matrix = self.final_hidden_uni_matrix[:, :, :, feature_num]
            final_matrix =  self.final_uni_matrix[:, :, :, feature_num]
            all_hidden_states_single_variable = individual_all_hidden_states[:, :, :, feature_num]
            final_hidden_state = individual_last_hidden_state[:, :, :, feature_num].permute(0, 2, 1).contiguous()

            mat1 = torch.bmm(all_hidden_states_single_variable, attention_matrix_uni)
            mat2 = torch.bmm(mat1, final_hidden_state)
            attention_score = torch.sigmoid(mat2)

            context_vector_individual = attention_score * all_hidden_states_single_variable
            context_vector_individual = torch.sum(context_vector_individual, dim=1)
            context_vector_individual = context_vector_individual.view(context_vector_individual.size(0), context_vector_individual.size(1), 1)


            attended_states = torch.bmm(context_vector_matrix_uni, context_vector_individual)
            h_intermediate1 = attended_states + torch.bmm(hidden_matrix, final_hidden_state)

            if (feature_num == 0):
                h_output = torch.bmm(final_matrix, h_intermediate1)
            else:
                h_output += torch.bmm(final_matrix, h_intermediate1)

        h_intermediate2 = torch.bmm(self.bridge_matrix, h_output)

        """
           Combining the two
        """
        h_intermediate = h_intermediate + h_intermediate2
        result = torch.bmm(self.final_matrix, h_intermediate)
        result = result.permute(0, 2, 1).contiguous()
        result = result.squeeze()

        """
           Remove from result the extra result points which were added as a result of padding
        """
        final_result = result[:result.size(0) - diff]

        """
        Adding highway network to it
        """

        if (self.hw > 0):
            z = x[:, -self.hw:, :];
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw);
            z = self.highway(z);
            z = z.view(-1, self.original_columns);
            res = final_result + z;

        return torch.sigmoid(res)

class TPA_LSTM(nn.Module):
    def __init__(self, args, data):
        super(TPA_LSTM, self).__init__()
        self.use_cuda = args.cuda
        self.window_length = args.window;  # window, read about it after understanding the flow of the code...What is window size? --- temporal window size (default 24 hours * 7)
        self.original_columns = data.original_columns  # the number of columns or features
        self.hidR = args.hidRNN;
        self.hidden_state_features = args.hidden_state_features
        self.hidC = args.hidCNN;
        self.hidS = args.hidSkip;
        self.Ck = args.CNN_kernel;  # the kernel size of the CNN layers
        self.skip = args.skip;
        self.pt = (self.window_length - self.Ck) // self.skip
        self.hw = args.highway_window
        self.num_layers_lstm = args.num_layers_lstm
        self.lstm = nn.LSTM(input_size=self.original_columns, hidden_size=self.hidden_state_features,
                            num_layers=self.num_layers_lstm,
                            bidirectional=False);
        self.compute_convolution = nn.Conv2d(1, self.hidC, kernel_size=(
            self.Ck, self.hidden_state_features))  # hidC are the num of filters, default value of Ck is one
        self.attention_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.hidC, self.hidden_state_features, requires_grad=True, device='cuda'))
        self.context_vector_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.hidden_state_features, self.hidC, requires_grad=True, device='cuda'))
        self.final_state_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.hidden_state_features, self.hidden_state_features, requires_grad=True, device='cuda'))
        self.final_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.original_columns, self.hidden_state_features, requires_grad=True, device='cuda'))
        torch.nn.init.xavier_uniform(self.attention_matrix)
        torch.nn.init.xavier_uniform(self.context_vector_matrix)
        torch.nn.init.xavier_uniform(self.final_state_matrix)
        torch.nn.init.xavier_uniform(self.final_matrix)
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.original_columns));  # kernel size is size for the filters
        self.GRU1 = nn.GRU(self.hidC, self.hidR);
        self.dropout = nn.Dropout(p=args.dropout);
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS);
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.original_columns);
        else:
            self.linear1 = nn.Linear(self.hidR, self.original_columns);
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1);
        self.output = None;
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;

    def forward(self, input):
        batch_size = input.size(0);
        if (self.use_cuda):
            x = input.cuda()

        """
           Step 1. First step is to feed this information to LSTM and find out the hidden states 

            General info about LSTM:

            Inputs: input, (h_0, c_0)
        - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
          of the input sequence.
          The input can also be a packed variable length sequence.
          See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
          :func:`torch.nn.utils.rnn.pack_sequence` for details.
        - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial hidden state for each element in the batch.
          If the RNN is bidirectional, num_directions should be 2, else it should be 1.
        - **c_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial cell state for each element in the batch.

          If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.


    Outputs: output, (h_n, c_n)
        - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
          containing the output features `(h_t)` from the last layer of the LSTM,
          for each t. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
          given as the input, the output will also be a packed sequence.

          For the unpacked case, the directions can be separated
          using ``output.view(seq_len, batch, num_directions, hidden_size)``,
          with forward and backward being direction `0` and `1` respectively.
          Similarly, the directions can be separated in the packed case.
        - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the hidden state for `t = seq_len`.

          Like *output*, the layers can be separated using
          ``h_n.view(num_layers, num_directions, batch, hidden_size)`` and similarly for *c_n*.
        - **c_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the cell state for `t = seq_len`

        """
        input_to_lstm = x.permute(1, 0, 2).contiguous()  # input to lstm is of shape (seq_len, batch, input_size) (x shape (batch_size, seq_length, features))
        lstm_hidden_states, (h_all, c_all) = self.lstm(input_to_lstm)
        hn = h_all[-1].view(1, h_all.size(1), h_all.size(2))

        """
            Step 2. Apply convolution on these hidden states. As in the paper TPA-LSTM, these filters are applied on the rows of the hidden state
        """
        output_realigned = lstm_hidden_states.permute(1, 0, 2).contiguous()
        hn = hn.permute(1, 0, 2).contiguous()
        # cn = cn.permute(1, 0, 2).contiguous()
        input_to_convolution_layer = output_realigned.view(-1, 1, self.window_length, self.hidden_state_features);
        convolution_output = F.relu(self.compute_convolution(input_to_convolution_layer));
        convolution_output = self.dropout(convolution_output);


        """
            Step 3. Apply attention on this convolution_output
        """
        convolution_output = convolution_output.squeeze(3)

        """
                In the next 10 lines, padding is done to make all the batch sizes as the same so that they do not pose any problem while matrix multiplication
                padding is necessary to make all batches of equal size
        """
        final_hn = torch.zeros(self.attention_matrix.size(0), 1, self.hidden_state_features)
        final_convolution_output = torch.zeros(self.attention_matrix.size(0), self.hidC, self.window_length)
        diff = 0
        if (hn.size(0) < self.attention_matrix.size(0)):
            final_hn[:hn.size(0), :, :] = hn
            final_convolution_output[:convolution_output.size(0), :, :] = convolution_output
            diff = self.attention_matrix.size(0) - hn.size(0)
        else:
            final_hn = hn
            final_convolution_output = convolution_output

        """
           final_hn, final_convolution_output are the matrices to be used from here on
        """
        convolution_output_for_scoring = final_convolution_output.permute(0, 2, 1).contiguous()
        final_hn_realigned = final_hn.permute(0, 2, 1).contiguous()
        convolution_output_for_scoring = convolution_output_for_scoring.cuda()
        final_hn_realigned = final_hn_realigned.cuda()
        mat1 = torch.bmm(convolution_output_for_scoring, self.attention_matrix).cuda()
        scoring_function = torch.bmm(mat1, final_hn_realigned)
        alpha = torch.nn.functional.sigmoid(scoring_function)
        context_vector = alpha * convolution_output_for_scoring
        context_vector = torch.sum(context_vector, dim=1)

        """
           Step 4. Compute the output based upon final_hn_realigned, context_vector
        """
        context_vector = context_vector.view(-1, self.hidC, 1)
        h_intermediate = torch.bmm(self.final_state_matrix, final_hn_realigned) + torch.bmm(self.context_vector_matrix, context_vector)
        result = torch.bmm(self.final_matrix, h_intermediate)
        result = result.permute(0, 2, 1).contiguous()
        result = result.squeeze()

        """
           Remove from result the extra result points which were added as a result of padding 
        """
        final_result = result[:result.size(0) - diff]

        """
        Adding highway network to it
        """

        if (self.hw > 0):
            z = x[:, -self.hw:, :];
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw);
            z = self.highway(z);
            z = z.view(-1, self.original_columns);
            res = final_result + z;

        return torch.sigmoid(res)

"""THE DRIVER CLASS TO RUN THIS CODE"""

"""FUTURE SCOPE, ADD ARGUMENTS AS NEEDED"""



parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default="exp/data/VN30_price.csv",
                    help='location of the data file')
#, required=True
parser.add_argument('--model', type=str, default='TPA_LSTM_Modified',
                    help='')
parser.add_argument('--hidden_state_features', type=int, default=12,
                    help='number of features in LSTMs hidden states')
parser.add_argument('--num_layers_lstm', type=int, default=1,
                    help='num of lstm layers')
parser.add_argument('--hidden_state_features_uni_lstm', type=int, default=1,
                    help='number of features in LSTMs hidden states for univariate time series')
parser.add_argument('--num_layers_uni_lstm', type=int, default=1,
                    help='num of lstm layers for univariate time series')
parser.add_argument('--attention_size_uni_lstm', type=int, default=10,
                    help='attention size for univariate lstm')
parser.add_argument('--hidCNN', type=int, default=10,
                    help='number of CNN hidden units')
parser.add_argument('--hidRNN', type=int, default=100,
                    help='number of RNN hidden units')
parser.add_argument('--window', type=int, default=24 * 7,
                    help='window size')
parser.add_argument('--CNN_kernel', type=int, default=1,
                    help='the kernel size of the CNN layers')
parser.add_argument('--highway_window', type=int, default=24,
                    help='The window size of the highway component')
parser.add_argument('--clip', type=float, default=10.,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=3000,
                    help='upper epoch limit') #30
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=54321,
                    help='random seed')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model/model.pt',
                    help='path to save the final model')
parser.add_argument('--cuda', type=str, default=False)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.5)
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--skip', type=float, default=24)
parser.add_argument('--hidSkip', type=int, default=5)
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--output_fun', type=str, default='sigmoid')
args = parser.parse_args()

def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval();
    total_loss = 0;
    total_loss_l1 = 0;
    n_samples = 0;
    predict = None;
    test = None;

    for X, Y in data.get_batches(X, Y, batch_size, False):
        output = model(X);
        if predict is None:
            predict = output;
            test = Y;
        else:
            predict = torch.cat((predict, output));
            test = torch.cat((test, Y));

        scale = data.scale.expand(output.size(0), data.original_columns)
        total_loss += evaluateL2(output * scale, Y * scale).data
        total_loss_l1 += evaluateL1(output * scale, Y * scale).data
        n_samples += (output.size(0) * data.original_columns);
    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    predict = predict.data.cpu().numpy();
    Ytest = test.data.cpu().numpy();

    #print(predict.shape, Ytest.shape)

    sigma_p = (predict).std(axis=0);
    sigma_g = (Ytest).std(axis=0);
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0);
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g);
    correlation = (correlation[index]).mean();
    return rse, rae, correlation;


def train(data, X, Y, model, criterion, optim, batch_size):  # X is train set, Y is validation set, data is the whole data
    model.train();
    total_loss = 0;
    n_samples = 0;
    for X, Y in data.get_batches(X, Y, batch_size, True):
        #print(Y)
        model.zero_grad();
        output = model(X);
        scale = data.scale.expand(output.size(0), data.original_columns)
        loss = criterion(output * scale, Y * scale);
        loss.backward();
        grad_norm = optim.step();
        total_loss += loss.data;
        n_samples += (output.size(0) * data.original_columns);
    return total_loss / n_samples
    return 1




Data = Data_utility(args.data, 0.6, 0.2, args.cuda, args.horizon, args.window, args.normalize); #SPLITS THE DATA IN TRAIN AND VALIDATION SET, ALONG WITH OTHER THINGS, SEE CODE FOR MORE
print(Data.rse);

device = 'cpu'


model = eval(args.model)(args, Data);
if(args.cuda):
    model.cuda()


#print(dict(model.named_parameters()))
if args.L1Loss:
    criterion = nn.L1Loss(size_average=False);
else:
    criterion = nn.MSELoss(size_average=False);
evaluateL2 = nn.MSELoss(size_average=False);
evaluateL1 = nn.L1Loss(size_average=False)
if args.cuda:
    criterion = criterion.cuda()
    evaluateL1 = evaluateL1.cuda();
    evaluateL2 = evaluateL2.cuda();

nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)

#print(list(model.parameters())[0].grad)
list(model.parameters())
#optim = Optim.Optim(model.parameters(), args.optim, args.lr, args.clip,)
optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)  #.01 1e-05
best_val = 10000000;

try:
    print('begin training');
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
        #print(train_loss)
        val_loss, val_rae, val_corr = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1, args.batch_size);
        print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(
                epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr))
        # Save the model if the validation loss is the best we've seen so far.

        if val_loss < best_val:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val = val_loss
        if epoch % 5 == 0:
            test_acc, test_rae, test_corr = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
                                                     args.batch_size);
            print("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')