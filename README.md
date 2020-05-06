# RLComGen

RLCodeGen is a project using reinforcement learning algorithm Actor-Critic to generate summarization for code snippet.
The source code and dataset are opened.

# Requirements

Python 3.7

Pytorch 1.4

# Config

Every model has its own `config.py` file in the model folder, we can change the configuration in this file.

# Model Training

Pretrain the actor of RLCom: `python main.py --method RLCom --pretrain actor`

Pretrain the critic of RLCom: `python main.py --method RLCom --pretrain critic`

Train the RLCom: `python main.py --method RLCom`

Train the deepcom: `python main.py --method deepcom`

Train the TL-CodeSum: `python main.py --method TLCodeSum`

RLComAPI is the model RLCom with transferred api knowledge, that's to say, is using TL-CodeSum as actor and critic in Actor-Critic algorithm. It is the same usage with RLCom and RLComAPI.

In every training epoch, if the model has the greater performance, this model will be saved in "[model_name]/save/ast_[training_epoch].pkl".

Continue to train the model starting from a checkpoint: `python main.py --method [model_name] --start [training_epoch]`

# Translation

`python translate.py --method [model_name] --start [training_epoch]`

This instruction translates the test code snippets into their summarizations. The result is saved in "pred.txt".