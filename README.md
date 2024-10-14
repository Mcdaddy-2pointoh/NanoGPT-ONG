# Description
An attempt to follow Andrej Karapathy's [Let's build GPT: from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY). In order to better understand "Attention is all you need" and the transformers architecture that is the corner stone of GenAI. The project is split into multiple parts Given below.

## Project Documentation
1. Data Loaders: This file available at `./utils/loaders.py` is a module containing all possible data loaders to read and load data from rich text formats

2. Data Augmenters: This file available at `./utils/loaders.py` is a module containing all possible data augmenters needed to convert data from rich text format to model suitable and machine readable format. This module includes:
    i. Tokenizers
    ii. Batch generators
    iii. Train test splitters

3. Project requirements are defined in `./requirements.txt`