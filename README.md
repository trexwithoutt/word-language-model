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

### Serve model through REST API

FloydHub supports seving mode for demo and testing purpose. Before serving your model through REST API,
you need to create a `floyd_requirements.txt` and declare the flask requirement in it. If you run a job
with `--mode serve` flag, FloydHub will run the `app.py` file in your project
and attach it to a dynamic service endpoint:

```bash
floyd run --gpu --mode serve --env pytorch-0.2  --data <USERNAME>/dataset/<PENN-TB3>/<VERSION>:input --data <REPLACE_WITH_JOB_OUTPUT_NAME>:model
```

The above command will print out a service endpoint for this job in your terminal console.

The service endpoint will take a couple minutes to become ready. Once it's up, you can interact with the model by sending a POST request wih the number of words and the temperature that the model will use to generate text:
```bash
# Template
# curl -X POST -o <NAME_&_PATH_DOWNLOADED_GENERATED_TEXT> -F "words=<NUMBER_OF_WORDS_TO_GENERATE>" -F "temperature=<TEMPERATURE>" <SERVICE_ENDPOINT>

curl -X POST -o generated.txt -F "words=100" -F "temperature=3" https://www.floydlabs.com/expose/vk47ixT8NeYBTFeMavbWta
```

Any job running in serving mode will stay up until it reaches maximum runtime. So
once you are done testing, **remember to shutdown the job!**

*Note that this feature is in preview mode and is not production ready yet*

## More resources

Some useful resources on NLP for Deep Learning and language modeling task:

- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Natural Language Processing with Deep Learning - Stanford](https://youtu.be/OQQ-W_63UgQ)
- [Oxford Deep NLP 2017 course](https://github.com/oxford-cs-deepnlp-2017/lectures)
