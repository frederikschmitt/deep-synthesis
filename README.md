# DeepSynthesis
Implementation and datasets of my master thesis on "LTL Synthesis from Specification Patterns with Neural Networks"

## Docker

We recommend to run the data generation and experiments in a Docker container because of the dependencies of this project. More information on Docker on how to install Docker on your machine can be found [here](https://docs.docker.com). To pull the Docker image from [Docker Hub](https://hub.docker.com) run:

`docker pull frederikschmitt/deepsynthesis:latest-cpu`

## Date Generation

### SYNTCOMP

To generate a dataset based on SYNTCOMP patterns run the `data_generation_guarantees` script and provide the path to the file with the SYNTCOMP patterns and a directory where the dataset is saved:

```python -m deepsynthesis.data.synthesis.data_generation.data_generation_guarantees --guarantees /deep-synthesis/data/syntcomp-patterns.json --data-dir /deep-synthesis/data```

To list all available options of the data generation run the script with `--help`.

### Grammar

To generate a dataset based on handcrafted specification patterns that were designed using a template grammar run the `data_generation_grammar` script and provide a directory where the dataset is saved:

```python -m deepsynthesis.data.synthesis.data_generation.data_generation_grammar --data-dir /deep-synthesis/data```

To list all available options of the data generation run the script with `--help`.

## Training

To train a Transformer on the LTL synthesis problem run the `synthesis_transformer_experiment` script with a specific dataset and hyperparameters of choice. For example, to train a Transformer on dataset **SC100** for 3 epochs run:

```python -m deepsynthesis.experiments.synthesis_transformer_experiment --dataset /deep-synthesis/data/SC100 --epochs 3```

For an overview of the available hyperparamter options run the script with `--help`.