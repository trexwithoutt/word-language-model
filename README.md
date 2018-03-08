# Word-level language modeling RNN

This example trains a multi-layer RNN (Quasi-RNN, GRU, or LSTM) on a language modeling task.
By default, the training script uses the PTB dataset, provided.
The trained model can then be used by the generate script to generate new text.
This is a porting of [pytorch/examples/word_language_model](https://github.com/pytorch/examples/tree/master/word_language_model) making it usables on [FloydHub](https://www.floydhub.com/).

## Usage

The `main.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help         show this help message and exit
  --data DATA        location of the data corpus
  --model MODEL      type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)
  --emsize EMSIZE    size of word embeddings
  --nhid NHID        number of hidden units per layer
  --nlayers NLAYERS  number of layers
  --lr LR            initial learning rate
  --optlr            learning rate for optimizer
  --clip CLIP        gradient clipping
  --epochs EPOCHS    upper epoch limit
  --batch-size N     batch size
  --adasoft          activate adaptive softmax
  --bptt BPTT        sequence length
  --pre              pre-trained weight (200 or 300 emsize if using)
  --dropout DROPOUT  dropout applied to layers (0 = no dropout)
  --decay DECAY      learning rate decay per epoch
  --tied             tie the word embedding and softmax weights
  --seed SEED        random seed
  --cuda             use CUDA
  --log-interval N   report interval
  --save SAVE        path to save the final model
```

With these arguments, a variety of models can be tested.
As an example, the following arguments produce slower but better models:

```bash
python main.py --cuda --emsize 300 --nhid 300 --dropout 0.2 --epochs 5           # Test perplexity of 98.73
```

These perplexities are equal or better than
[Recurrent Neural Network Regularization (Zaremba et al. 2014)](https://arxiv.org/pdf/1409.2329.pdf)
and are similar to [Using the Output Embedding to Improve Language Models (Press & Wolf 2016](https://arxiv.org/abs/1608.05859) and [Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling (Inan et al. 2016)](https://arxiv.org/pdf/1611.01462.pdf), though both of these papers have improved perplexities by using a form of recurrent dropout [(variational dropout)](http://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks).


## Run on FloydHub

Here's the commands to training, evaluating and serving your language modeling task on FloydHub.

### Project Setup

Before you start, log in on FloydHub with the [floyd login](http://docs.floydhub.com/commands/login/) command, then fork and init the project:

```bash
$ git clone https://github.com/trexwithoutt/word-language-model.git
$ cd word-language-model
```

## More resources

Some useful resources on NLP for Deep Learning and language modeling task:

- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Natural Language Processing with Deep Learning - Stanford](https://youtu.be/OQQ-W_63UgQ)
- [Oxford Deep NLP 2017 course](https://github.com/oxford-cs-deepnlp-2017/lectures)
