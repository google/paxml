# Paxml (aka Pax)

Pax is a framework to configure and run machine learning experiments on top of Jax.
## Quickstart
### Setting up a Cloud TPU VM

We refer to
[this page](https://cloud.google.com/tpu/docs/users-guide-tpu-vm#managing_tpus)
for more exhaustive documentation about starting a Cloud TPU project. The
following command is sufficient to create a Cloud TPU VM with 8 cores from a
corp machine.

```bash
export ZONE=us-central2-b
export VERSION=tpu-vm-v4-base
export PROJECT=<your-project>
export ACCELERATOR=v4-8
export TPU_NAME=paxml

#create a TPU VM
gcloud compute tpus tpu-vm create $TPU_NAME --zone=$ZONE --version=$VERSION --project=$PROJECT --accelerator-type=$ACCELERATOR
```

The corresponding VM instance can then be accessed via ssh.

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE
```

### Installing Pax

After ssh-ing the VM, paxml can be installed using
[pip](https://pypi.org/project/pip/).

```bash
$ python3 -m pip install -U pip
$ python3 -m pip install paxml jax[tpu] \
-f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```
For the exact version of dependencies used to build/test each release, go to the corresponding release branch rX.Y.Z and check out `paxml/pip_package/requirements.txt`

### Run a test model
```bash
# example model using pjit (SPMD)
python3 .local/lib/python3.8/site-packages/paxml/main.py \
--exp=tasks.lm.params.lm_cloud.LmCloudSpmd2BLimitSteps \
--job_log_dir=gs://<your-bucket>

# example model using pmap
python3 .local/lib/python3.8/site-packages/paxml/main.py \
--exp=tasks.lm.params.lm_cloud.LmCloudTransformerAdamLimitSteps \
--job_log_dir=gs://<your-bucket> \
--pmap_use_tensorstore=True
```

# Data inputs

## Intro

Input is an instance of the `BaseInput`
class for getting data into model for train/eval/decode.

```python
class BaseInput:

  def get_next(self):
    pass

  def reset(self):
    pass
```

It acts like an iterator: `get_next()` returns a `NestedMap`, where each field
is a numerical array with batch size as its leading dimension.

Each input is configured by a subclass of `BaseInput.HParams`.
In this page, we use `p` to denote an instance of a `BaseInput.Params`, and it
instantiates to `input`.

## Multihost infeed

In Pax, data is always multihost: Each Jax process will have a separate,
independent `input` instantiated. Their params will have different
`p.infeed_host_index`, set automatically by Pax.

Hence, the local batch size seen on each host is `p.batch_size`, and the global
batch size is `(p.batch_size * p.num_infeed_hosts)`. One will often see
`p.batch_size` set to `jax.local_device_count() * PERCORE_BATCH_SIZE`.

Due to this multihost nature, `input` must be sharded properly.

For training, each `input` must never emit identical batches, and for eval on a
finite dataset, each `input` must terminate after the same number of batches.
The best solution is to have the input implementation properly shard the data,
such that each `input` on different hosts do not overlap. Failing that, one can
also use different random seed to avoid duplicate batches during training.

## Input for eval data

`input.reset()` is never called on training data, but it can for eval (or
decode) data.

For each eval (or decode) run, Pax will fetch `N` batches from `input` by
calling `input.get_next()` `N` times, after which Pax will optionally reset by
calling `input.reset()`, depending on the value of `p.reset_for_eval`.

The number of batches used, `N`, can be a fixed number specified by user, via
`p.eval_loop_num_batches`; or `N` can be dynamic
(`p.eval_loop_num_batches=None`), in which case we call `input.get_next()` until
we exhaust all of its data (by raising `StopIteration` or
`tf.errors.OutOfRange`).

|                          | `N`: static                  | `N`: dynamic       |
| ------------------------ | ---------------------------- | ------------------ |
| `p.reset_for_eval=True`  | Each eval run uses the first | One epoch per eval |
:                          : `N` batches consistently.    : run. `input` must  :
:                          : `p.eval_loop_num_batches=N`. : be finite and      :
:                          : Not supported yet.           : raise after its    :
:                          :                              : data is exhausted. :
:                          :                              : All shards must    :
:                          :                              : raise after the    :
:                          :                              : same number of     :
:                          :                              : batches.           :
| `p.reset_for_eval=False` | Each eval run uses           | Not supported.     |
:                          : non-overlapping `N` batches  :                    :
:                          : on a rolling basis.          :                    :
:                          : `p.eval_loop_num_batches=N`. :                    :
:                          : `input` must repeat          :                    :
:                          : indefinitely and never       :                    :
:                          : raise.                       :                    :

For the "eval on exactly one epoch" use case with `p.reset_for_eval=True,
p.eval_loop_num_batches=None`, input must handle sharding correctly such that
each shard raises at the same step after exactly the same number of batches are
produced. This usually means that the input must pad the eval data. This is done
automatically by `SeqIOInput` and `LingvoEvalAdaptor` (see more below).

### Eval metrics

For the majority of inputs, we only ever call `get_next()` on them to get
batches of data. One type of eval data is an exception to this, where "how to
compute metrics" is also defined on the input object as well.

This is only supported with `SeqIOInput` that defines some caonical eval
benchmark. Specifically, Pax uses `predict_metric_fns` and `score_metric_fns()` defined on the SeqIO task to compute
eval metrics (although Pax does not depend on SeqIO evaluator directly).

## Best practices

When a model uses multiple inputs, either between train/eval or different
training data between pretraining/finetuning, users must ensure that the
tokenizers used by the inputs are identical, especially when importing different
inputs implemented by others.

Users can sanity check the tokenizers by decoding some ids with
`input.ids_to_strings()`.

It's always a good idea to sanity check the data by looking at a few batches.
Users can easily reproduce the param in a colab and inspect the data:

```python
p = ... # specify the intended input param
inp = p.Instantiate()
b = inp.get_next()
print(b)
```

Training data typically should not use a fixed random seed. This is because if
the training job is preempted, training data will start to repeat itself. In
particular, for Lingvo inputs, we recommend setting `p.input.file_random_seed =
0` for training data.

To test for whether sharding is handled correctly, users can manually set
different values for `p.num_infeed_hosts, p.infeed_host_index` and see whether
the instantiated inputs emit different batches.

## Input types

Pax supports 3 types of inputs: SeqIO, Lingvo, and custom.

### SeqIO

`SeqIOInput` can be used to import datasets.

SeqIO inputs handle correct sharding and padding of eval data automatically.

### Lingvo

`LingvoInputAdaptor` can be used to import datasets.

The input is fully delegated to the Lingvo implementation, which may or may not
handle sharding automatically.

For GenericInput based Lingvo input implementation using a fixed
`packing_factor`, we recommend to use
`LingvoInputAdaptorNewBatchSize` to specify a bigger batch size for the inner Lingvo input and put the desired
(usually much smaller) batch size on `p.batch_size`.

For eval data, we recommend using
`LingvoEvalAdaptor` to handle sharding and padding for running eval over one epoch.

### Custom

Custom subclass of `BaseInput`. Users implement their own subclass, typically
with `tf.data` or SeqIO.

Users can also inherit an existing input class to only customize post processing
of batches. For example:

```python
class MyInput(base_input.LingvoInputAdaptor):

  def get_next(self):
    batch = super().get_next()
    # modify batch: batch.new_field = ...
    return batch
```

#Key Pax components:

## Hyperparameters

Hyperparameters are an important part of defining models and configuring
experiments.

To integrate better with Python tooling, Pax/Praxis uses a pythonic
dataclass based configuration style for hyperparameters.

```python
class Linear(base_layer.BaseLayer):
  """Linear layer without bias."""

  class HParams(BaseHParams):
    """Associated hyperparams for this layer class.

    Attributes:
      input_dims: Depth of the input.
      output_dims: Depth of the output.
    """
    input_dims: int = 0
    output_dims: int = 0
```

### Nesting

It's also possible to nest HParams dataclasses, in the example below, the
linear_tpl attribute is a nested Linear.HParams.

```python
class FeedForward(base_layer.BaseLayer):
  """Feedforward layer with activation."""

  class HParams(BaseHParams):
    """Associated hyperparams for this layer class.

    Attributes:
      input_dims: Depth of the input.
      output_dims: Depth of the output.
      has_bias: Adds bias weights or not.
      linear_tpl: Linear layer params.
      activation_tpl: Activation layer params.
    """
    input_dims: int = 0
    output_dims: int = 0
    has_bias: bool = True
    linear_tpl: BaseHParams = sub_config_field(Linear.HParams)
    activation_tpl: activations.BaseActivation.HParams = sub_config_field(
        ReLU.HParams)
```


## Layers

A Layer represents an arbitrary function possibly with trainable parameters. A
Layer can contain other Layers as children. Layers are the essential building
blocks of models. Layers inherit from the Flax nn.Module.

Typically layers define two methods:

### setup

This method creates trainable weights and child layers.

### fprop

This method defines the forward propagation function, computing some output
based on the inputs. Additionally, fprop might add summaries or track auxiliary
losses.

### Fiddle and Shared layers
Fiddle is an open-sourced Python-first configuration library
designed for ML applications. Pax/Praxis supports interoperability with Fiddle
Config/Partial(s) and some advanced features like eager error checking and
shared parameters.

```python
fdl_config = Linear.HParams.config(input_dims=1, output_dims=1)

# A typo.
fdl_config.input_dimz = 31337  # Raises an exception immediately to catch typos fast!


fdl_partial = Linear.HParams.partial(input_dims=1)
```

Using Fiddle, layers can be configured to be shared (eg: instantiated only once
with shared trainable weights).

## Model

A model defines solely the network, typically a collection of Layers and defines
interfaces for interacting with the model such as decoding, etc.

Some example base models
include:

*   LanguageModel
*   SequenceModel
*   ClassificationModel

## Task

A Task contains one more more Models and Learner/Optimizers. The simplest Task
subclass is a `SingleTask` which requires the following Hparams:

```python
  class HParams(base_task.BaseTask.HParams):
    """Task parameters.

    Attributes:
      name: Name of this task object, must be a valid identifier.
      model: The underlying JAX model encapsulating all the layers.
      train: HParams to control how this task should be trained.
      metrics: A BaseMetrics aggregator class to determine how metrics are
         computed.
      loss_aggregator: A LossAggregator aggregator class to derermine how the
        losses are aggregated (e.g single or MultiLoss)
      vn: HParams to control variational noise.
```

## Releases
PyPI Version | Commit
------------ | ----------------------------------------
0.1.0        | 546370f5323ef8b27d38ddc32445d7d3d1e4da9a



    Copyright 2022 Google LLC

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
