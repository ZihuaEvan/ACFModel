# ACF for parsing tree

## Dependencies

* python3
* pytorch 1.0

We use BERT tokenizer from [PyTorch-Transformers](https://github.com/huggingface/pytorch-transformers) to tokenize words. Please install [PyTorch-Transformers](https://github.com/huggingface/pytorch-transformers) following the instructions of the repository.  


## Training
For grammar induction training:  
```python3 main.py -train -model_dir [model_dir] -num_step 60000```  
The training file 'data/train.txt' includes all WSJ data except 'WSJ_22 and WSJ_23'.   

## Evaluation
For grammar induction testing:  
```python3 main.py -test -model_dir [model_dir]```  
The code creates a result directory named model_dir. The result directory includes 'bracket.json' and 'tree.txt'. File 'bracket.json' contains the brackets of trees outputted from the model and they can be used for evaluating F1. The ground truth brackets of testing data can be obtained by using code of [on-lstm](https://github.com/yikangshen/Ordered-Neurons). File 'tree.txt' contains the parse trees. The default testing file 'data/test.txt' contains the tests of wsj_23.   

## Acknowledgements
* This code is based on [Tree Transformer](https://github.com/yaushian/Tree-Transformer)， [Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html).  
* The code of BERT optimizer is taken from [PyTorch-Transformers](https://github.com/huggingface/pytorch-transformers).  

