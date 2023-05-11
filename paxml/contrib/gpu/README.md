# Pax on GPUs
This folder contains scripts that optimize Pax for GPUs.

## Building the Container
The Dockerfile in this folder contains all relevant dataset/gpu dependencies. Run the following command to build a container using this Dockerfile: `bash paxml/contrib/gpu/docker/build.sh <container_name>`. Be sure to run this command from the top-level `paxml` directory in the cloned repository.

The scripts in `scripts_gpu` have been validated with the following paxml commit: [75d70d7994507974311b29254617d39c9cd4764e](https://github.com/google/paxml/commit/75d70d7994507974311b29254617d39c9cd4764e). Note that this container is built to run paxml at the tested commit.

## Running the Container (single node)
Run the following command to launch a container interactively: `bash paxml/contrib/gpu/docker/interactive_pull_and_launch.sh <container_URL> <dataset_path> <vocab_path> <workspace_path>`, where `<workspace_path>` refers to the directory to be mounted to the container. This is where your experiment configs and run scripts should reside. Again, make sure this command is run from the top-level directory of the cloned paxml repository. 

## Downloading The Pile and Lambada Datasets
The scripts `scripts_gpu/download_the_pile.py` and `scripts_gpu/download_lambada.py` will download The Pile and the Lambada datasets to the `TFDS_DATA_DIR` enviroment variable. To control the location of the downloaded datasets, use the following command prior to running the download scripts: `export TFDS_DATA_DIR=<path_to_dowload_data_to>`. After the data has been successfully downloaded, use the same `TFDS_DATA_DIR` when running experiments.

## Running a Job
Note than when training with The Pile dataset, you must provide the `TFDS_DATA_DIR` as a command-line argument and a `VOCAB_PATH` (the path to a pretrained sentencepiece model used for tokenization) as an environment variable in your bash script (see below for examples). The sentencepiece model used in the following experiments is `gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model`. This model was trained using [these instructions](https://github.com/sgpyc/training/blob/paxml-llm-draft/large_language_model/paxml/utils/generate_spm.md). See [below](#Downloading-the-SentencePiece-Model) for information on downloading `gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model` from Google Cloud. 

### Quick Runs
#### Single Node
See `scripts_gpu/run_pile_singlenode.sh` for an example of training a 126m model on a single node using The Pile. Once inside of your container, this script can be run using the following command: 
``` 
bash paxml/contrib/gpu/scripts_gpu/run_pile_singlenode.sh <TFDS_DATA_DIR> <VOCAB_PATH> <LOGDIR>
```
where `TFDS_DATA_DIR` is the path to the downloaded datasets, `VOCAB_PATH` is the path to the pretrained SentencePiece `.model` file, and `LOGDIR` is the relative path of the directory to which to write checkpoints and logging information.

See `scripts_gpu/run_lambada_singlenode` for an example of running zero-shot evaluation on the 126m model using the Lambada dataset. Use the following command to run this script:
``` 
bash paxml/contrib/gpu/scripts_gpu/run_lambada_singlenode.sh <TFDS_DATA_DIR> <VOCAB_PATH> <LOGDIR>
```
`TFDS_DATA_DIR` should contain the path to the Lambada dataset and `LOGDIR` should match the `LOGDIR` from the pretraining run.

#### Multi Node
See `scripts_gpu/example_slurm_pile.sub` for an example slurm submit file that launches an 8 node run with a 126 million parameter GPT model. Note that this script must be edited with your slurm account information. 

To launch `example_slurm_pile.sub`, simply run the following command: 
```
sbatch paxml/contrib/gpu/scripts_gpu/example_slurm_pretrain_pile.sub
```
Note that hyperparameters currently cannot be overwritten from the command line. In order to change a training hyperparameter (e.g. number of nodes, number of layers, precision), add a new config to `scripts_gpu/configs.py` which inherits from the desired config and overwrites the relevant hyperparameters. See the [configs](#Configs) section below for more information about the provided base configs, and see `SmallPileTest` in `scripts_gpu/configs.py` for an example config that overwrites the number of GPUs from an existing base config. 

## Configs
We provide three "base" model configs in `scripts_gpu/configs.py`. The first is a 126 million parameter GPT model. Convergence using The Pile dataset has been verified with this model. See the table below for convergence results. 

The remaining configs are 5 billion and 175 billion parameter models. Both 5B and 175B are provided for benchmarking purposes and have not been thoroughly tested for convergence to date. Note that while 126M and 5B are trained on the Pile, 175B uses the [C4 dataset](https://github.com/google/paxml/blob/7656f4913885fc8e810423ed78b47a2ec77e9bbf/paxml/tasks/lm/params/c4.py#L149) by default. 

The table below describes current performance of the given configs. Experiments were run using NVIDIA DGX A100 (8x A100 80G) nodes. Note that Lambada accuracy reported corresponds to the best accuracy seen across the run.

| Size | #GPUs | BS / GPU | Sequences/Sec | Estimated Walltime (days) | Lambada Accuracy | Convergence Log |
| ---- | ----- | -------- | ------------- | ------------------------- | ---------------- | --------------- |
| 126M |  64    | 4        |   1689.6      |   1.1                     |        0.397 (± 0.012)     | [log](https://tensorboard.dev/experiment/RCroDLAUQzGUoudzqD1NmQ/) |
| 5B   | 256    | 16       |     426       |     4.2                   |       N/A        | N/A             |
| 175B | 768    | 24       |    33.6       |      39.7                 |    N/A           |  N/A           |

Note: Estimated walltime is computed assuming full throughput continuously. In practice, true walltime may be greater due to compilation overheads and checkpointing. Linked convergence logs were not necessarily done with the topology described in `configs.py` and may have different walltimes, but the configs provided are the most performant configs tested. The throughput for these performant configs is reported in the table above.

**NOTE**: Current versions of Paxml are known to cause training instability and NaN loss with some configs. We are actively working on fixing this and will update this page once the issue is resolved.

## Downloading the SentencePiece Model
First, make sure you have the [Google Clould SDK](https://cloud.google.com/sdk/docs/install) installed. Next, log in to the Cloud using the following command: `gcloud auth login` and following the prompts. Once logged in, use the following command to download the vocab file to your current working directory: 
```
gsutil -m cp -r gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model .
```
