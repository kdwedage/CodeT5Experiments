# CodeT5Experiments

To run the code please make sure to install the required python packages, including: tree-sitter and tensorboard. Follow the installation steps in the CodeT5 directory.

## Running additional experiments:

There are three additional pretraining tasks, explored in this project:

- pretrain0 (Code+DFG+AST DAE): Pretrain with corrupted input: Code + DFG + AST, output is uncorrupted.
- pretrain1 (Full AST -> Code): Pretrain with Full AST input, output is original code.
- pretrain2 (Code+DFG DAE): Pretrain with corrupted input: Code + DFG, output is uncorrupted.

There are four additional finetuning tasks:

- finetune0: Single input: Code + DFG + AST, output: NL summary
- finetune1: Single input: Full AST, output: NL summary
- finetune2: Single input: Code + DFG, output: NL summary
- finetune3: Multiple inputs: (Code, Code + DFG, Full AST), output: NL summary

To run with the additional tasks, just change the task to one of 'pretrain0, pretrain1, pretrain2, finetune0, finetune1, finetune2, finetune3'.

Then run the command: python run\_exp.py --model\_tag codet5\_base --task TASK --sub\_task python
