# SLT

```
usage: __main__.py [-h] [-j JOB_ID] [-o OUTPUT_PATH] [-i INPUT_PATH]
                   [--action {build_sentence_dataset,build_piece_dataset,build_sentence_piece_dataset,train}] [--model-name MODEL_NAME]
                   [--seed SEED] [--verbosity VERBOSITY] [--wandbosity WANDBOSITY]

optional arguments:
  -h, --help            show this help message and exit
  -j JOB_ID, --job-id JOB_ID
                        The job id (and the name of the wandb group)
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        The directory which will contain saved model checkpoints
  -i INPUT_PATH, --input-path INPUT_PATH
                        The path to the directory containing the data to use (default: ./data)
  --action {build_sentence_dataset,build_piece_dataset,build_sentence_piece_dataset,train}
                        The action to perform (default: build_sentence_dataset)
  --model-name MODEL_NAME
                        The model name to use (default: gpt2)
  --seed SEED           The seed for torch, numpy, and python randomness (default: 1234)
  --verbosity VERBOSITY
                        The terminal output verbosity level (0 is min, 2 is max, default: 2)
  --wandbosity WANDBOSITY
                        The level of verbosity for wandb (0 is min, 2 is max, default: 2)
```