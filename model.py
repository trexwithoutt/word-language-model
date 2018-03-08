import torch.nn as nn
from torch.autograd import Variable
import math
from collections import namedtuple
import torch
from math import sqrt
from qrnn import *

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5,
                 tie_weights=False, adasoft=True, cutoff=[2000, 10000],
                 cuda=False, pre_trained=False, glove=None):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.glove = glove
        self.pre_trained = pre_trained
        self.adaptive_softmax = adasoft
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        elif rnn_type == 'QRNN':
            self.rnn = QRNN(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.LSTM(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        if adasoft:
            self.decoder = AdaptiveSoftmax(nhid, [*cutoff, ntoken+1],cuda)
        else:
            self.decoder = nn.Linear(nhid, ntoken+1)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.drop = nn.Dropout(dropout)
        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        if self.pre_trained:
            self.encoder.weight = nn.Parameter(self.glove)
        else:
            initrange = 0.1
            self.encoder.weight.data.uniform_(-initrange, initrange)
        
            if not self.adaptive_softmax:
                nn.init.xavier_normal(self.decoder.weight)
                self.decoder.bias.data.fill_(0)
            #self.decoder.bias.data.fill_(0)
            #self.decoder.weight.data.uniform_(-initrange, initrange)
        

    def forward(self, input, hidden, target=None):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        if self.adaptive_softmax:
            self.decoder.set_target(target.data) 
        decoded =self.decoder(output.contiguous() \
                              .view(output.size(0)*output.size(1), output.size(2)))
        #x = nn.functional.log_softmax(decoded)
        return decoded, hidden  #decoded.view(output.size(0), output.size(1), decoded.size(1))
    
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
    
    def log_prob(self, input, hidden, target):
        emb = self.encoder(input)
        output, hidden = self.rnn(emb, hidden)
        decoded = self.decoder.log_prob(output.contiguous() \
                .view(output.size(0) * output.size(1), output.size(2)))

        return decoded, hidden
    def reset(self):
        self.rnn.reset()
################################################  Adaptive Softmax  #############################################################

class AdaptiveSoftmax(nn.Module):
    def __init__(self, ninp, cutoff, cuda):
        super().__init__()

        self.ninp = ninp
        self.cutoff = cutoff
        self.nout = cutoff[0] + len(cutoff) - 1
        self.cuda = cuda

        self.head = nn.Linear(ninp, self.nout)
        self.tail = nn.ModuleList()

        for i in range(len(cutoff) - 1):
            seq = nn.Sequential(
                nn.Linear(ninp, ninp // 4 ** i, False),
                nn.Linear(ninp // 4 ** i, cutoff[i + 1] - cutoff[i], False)
            )

            self.tail.append(seq)

    def reset(self):
        std = 0.1
        #self.head.weight.data.uniform_(-std, std)
        nn.init.xavier_normal(self.head.weight)
        for tail in self.tail:
            nn.init.xavier_normal(tail[0].weight)
            nn.init.xavier_normal(tail[1].weight)
#             nn.init.kaiming_normal(tail[0].weight)
#             nn.init.kaiming_normal(tail[1].weight)
#             tail[0].weight.data.uniform_(-std, std)
#             tail[1].weight.data.uniform_(-std, std)

    def set_target(self, targets):
        self.id = []
        for i in range(len(self.cutoff) - 1):
            mask = targets.ge(self.cutoff[i]).mul(targets.lt(self.cutoff[i + 1]))

            if mask.sum() > 0:
                self.id.append(Variable(mask.float().nonzero().squeeze(1)))

            else:
                self.id.append(None)

    def forward(self, input):
        output = [self.head(input)]

        for i in range(len(self.id)):
            if self.id[i] is not None:
                output.append(self.tail[i](input.index_select(0, self.id[i])))

            else:
                output.append(None)

        return output

    def log_prob(self, input):
        if self.cuda:
            lsm = LogSoftmax().cuda()
        else:
            lsm = nn.LogSoftmax()
        head_out = self.head(input)
        batch_size = head_out.size(0)
        if self.cuda:
            prob = torch.zeros(batch_size, self.cutoff[-1]).cuda()
        else:
            prob = torch.zeros(batch_size, self.cutoff[-1])
        lsm_head = lsm(head_out)
        prob.narrow(1, 0, self.nout).add_(lsm_head.narrow(1, 0, self.nout).data)
        for i in range(len(self.tail)):
            pos = self.cutoff[i]
            i_size = self.cutoff[i + 1] - pos
            buffer = lsm_head.narrow(1, self.cutoff[0] + i, 1)
            buffer = buffer.expand(batch_size, i_size)
            lsm_tail = lsm(self.tail[i](input))
            prob.narrow(1, pos, i_size).copy_(buffer.data).add_(lsm_tail.data)

        return prob

class AdaptiveLoss(nn.Module):
    def __init__(self, cutoff):
        super().__init__()
        self.cutoff = cutoff
        self.criterion = nn.CrossEntropyLoss(size_average=False)
    def remap_target(self, targets):
        new_target = [targets.clone()]

        for i in range(len(self.cutoff) - 1):
            mask = targets.ge(self.cutoff[i]).mul(targets.lt(self.cutoff[i + 1]))
            new_target[0][mask] = self.cutoff[0] + i

            if mask.sum() > 0:
                new_target.append(targets[mask].add(-self.cutoff[i]))

            else:
                new_target.append(None)
        return new_target

    def forward(self, input, targets):
        batch_size = input[0].size(0)
        targets = self.remap_target(targets.data)

        output = 0.0

        for i in range(len(input)):
            if input[i] is not None:
                assert(targets[i].min() >= 0 and targets[i].max() <= input[i].size(1))
                output += self.criterion(input[i], Variable(targets[i]))

        output /= batch_size

        return output
    
################################################  Quasi-RNN  #############################################################

__author__ = '''salesforce'''


class CPUForgetMult(torch.nn.Module):
    def __init__(self):
        super(CPUForgetMult, self).__init__()

    def forward(self, f, x, hidden_init=None):
        result = []
        ###
        forgets = f.split(1, dim=0)
        prev_h = hidden_init
        for i, h in enumerate((f * x).split(1, dim=0)):
            if prev_h is not None: h = h + (1 - forgets[i]) * prev_h
            # h is (1, batch, hidden) when it needs to be (batch_hidden)
            # Calling squeeze will result in badness if batch size is 1
            h = h.view(h.size()[1:])
            result.append(h)
            prev_h = h
        return torch.stack(result)

class ForgetMult(torch.nn.Module):
    def __init__(self):
        super(ForgetMult, self).__init__()

    def forward(self, f, x, hidden_init=None):
        if hidden_init is None:
            return CPUForgetMult()(f, x)
        else:
            return CPUForgetMult()(f, x, hidden_init)

class QRNNLayer(nn.Module):
    '''https://github.com/salesforce/pytorch-qrnn/blob/master/torchqrnn/qrnn.py'''

    def __init__(self, input_size, hidden_size=None, save_prev_x=False, zoneout=0, window=1, output_gate=True):
        super(QRNNLayer, self).__init__()

        assert window in [1, 2], "This QRNN implementation currently only handles convolutional window of size 1 or size 2"
        self.window = window
        self.input_size = input_size
        self.hidden_size = hidden_size if hidden_size else input_size
        self.zoneout = zoneout
        self.save_prev_x = save_prev_x
        self.prevX = None
        self.output_gate = output_gate

        # One large matmul with concat is faster than N small matmuls and no concat
        self.linear = nn.Linear(self.window * self.input_size, 3 * self.hidden_size if self.output_gate else 2 * self.hidden_size)

    def reset(self):
        # If you are saving the previous value of x, you should call this when starting with a new state
        self.prevX = None

    def forward(self, X, hidden=None):
        seq_len, batch_size, _ = X.size()

        source = None
        if self.window == 1:
            source = X
        elif self.window == 2:
            # Construct the x_{t-1} tensor with optional x_{-1}, otherwise a zeroed out value for x_{-1}
            Xm1 = []
            Xm1.append(self.prevX if self.prevX is not None else X[:1, :, :] * 0)
            # Note: in case of len(X) == 1, X[:-1, :, :] results in slicing of empty tensor == bad
            if len(X) > 1:
                Xm1.append(X[:-1, :, :])
            Xm1 = torch.cat(Xm1, 0)
            # Convert two (seq_len, batch_size, hidden) tensors to (seq_len, batch_size, 2 * hidden)
            source = torch.cat([X, Xm1], 2)

        # Matrix multiplication for the three outputs: Z, F, O
        Y = self.linear(source)
        # Convert the tensor back to (batch, seq_len, len([Z, F, O]) * hidden_size)
        if self.output_gate:
            Y = Y.view(seq_len, batch_size, 3 * self.hidden_size)
            Z, F, O = Y.chunk(3, dim=2)
        else:
            Y = Y.view(seq_len, batch_size, 2 * self.hidden_size)
            Z, F = Y.chunk(2, dim=2)
        ###
        Z = torch.nn.functional.tanh(Z)
        F = torch.nn.functional.sigmoid(F)

        # If zoneout is specified, we perform dropout on the forget gates in F
        # If an element of F is zero, that means the corresponding neuron keeps the old value
        if self.zoneout:
            if self.training:
                mask = Variable(F.data.new(*F.size()).bernoulli_(1 - self.zoneout), requires_grad=False)
                F = F * mask
            else:
                F *= 1 - self.zoneout

        # Ensure the memory is laid out as expected for the CUDA kernel
        # This is a null op if the tensor is already contiguous
        Z = Z.contiguous()
        F = F.contiguous()
        # The O gate doesn't need to be contiguous as it isn't used in the CUDA kernel

        # Forget Mult
        # For testing QRNN without ForgetMult CUDA kernel, C = Z * F may be useful
        C = ForgetMult()(F, Z, hidden)

        # Apply (potentially optional) output gate
        if self.output_gate:
            H = torch.nn.functional.sigmoid(O) * C
        else:
            H = C

        # In an optimal world we may want to backprop to x_{t-1} but ...
        if self.window > 1 and self.save_prev_x:
            self.prevX = Variable(X[-1:, :, :].data, requires_grad=False)

        return H, C[-1:, :, :]


class QRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False, layers=None, **kwargs):
        assert bidirectional == False, 'Bidirectional QRNN is not yet supported'
        assert batch_first == False, 'Batch first mode is not yet supported'
        assert bias == True, 'Removing underlying bias is not yet supported'

        super(QRNN, self).__init__()

        self.layers = torch.nn.ModuleList(layers if layers else [QRNNLayer(input_size if l == 0 else hidden_size, hidden_size, **kwargs) for l in range(num_layers)])

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = len(layers) if layers else num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

    def reset(self):
        r'''If your convolutional window is greater than 1, you must reset at the beginning of each new sequence'''
        [layer.reset() for layer in self.layers]

    def forward(self, input, hidden=None):
        next_hidden = []

        for i, layer in enumerate(self.layers):
            input, hn = layer(input, None if hidden is None else hidden[i])
            next_hidden.append(hn)

            if self.dropout != 0 and i < len(self.layers) - 1:
                input = torch.nn.functional.dropout(input, p=self.dropout, training=self.training, inplace=False)

        next_hidden = torch.cat(next_hidden, 0).view(self.num_layers, *next_hidden[0].size()[-2:])

        return input, next_hidden



