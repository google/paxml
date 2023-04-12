# Models

doc/pax/models

[TOC]

## Introduction

A **model** is a collection of layers defining the network.

In Pax, Models are used in [Tasks][tasks], which are part of
[Experiments][experiments] that can be *trained*, *evaluated*, and *decoded* (as
well as a mix of these).

> Tip: For a rudamentary introduction to the basic Pax components, check out
> [Pax Elements][pax-elements]. If you want to dive in for a hands-on
> experience, try the [Pax Model and Task Jupyter Notebook][model_ipynb].

## Model How-To's

### Define a Model

In Pax, a Model inherits from `BaseModel`; and `BaseModel`, in turn, inherits
from `BaseLayer` along with a few interfaces for interacting with the
model:

*   `compute_predictions()`
*   `compute_loss()`
*   `decode()`
*   `process_decode_out()`

A BaseModel is a nothing more than a BaseLayer with a specific API that is used
to integrate with the Pax trainer:

```python
class BaseModel(base_layer.BaseLayer):
  ...
```

Therefore, to build your own Pax Model, you will need to define these methods in
your derived class.

#### `compute_predictions`

*Computes predictions for `input_batch`.*

The output can be in the form of probabilistic distributions, such as softmax
logits for discrete outputs, mixture of logistics for continuous values, or
regression values.

For training or evaluation, the output will be used for computing loss and
gradient updates, including comparing predicted distributions between teacher
and student for distillation. During inference the output can be used to compute
final outputs, perhaps with sampling.

Args:

*   input_batch: A `.NestedMap` object containing input tensors.

Returns:

*   Predictions, either a single Tensor, a `.NestedMap`, or a namedtuple.


#### `compute_loss`

*Computes the loss and other metrics for the given predictions.*

Args:

*   predictions: The output of `compute_predictions`.
*   input_batch: A `.NestedMap` object containing input tensors to this tower.

Returns:

*   WeightedScalars - A dict or NestedMap containing str keys and
    (value, weight) pairs as values, where one or more entries are
    expected to correspond to the loss (or losses).
*   A dict containing arbitrary tensors describing something about each
    training example, where the first dimension of each tensor is the batch
    index.
 
#### `decode`

*Decodes the input_batch.*

This code should be expected to run on TPUs.

Args:

*   input_batch: The input batch. A `NestedMap` of tensors. Or, if input batch
    splitting is used, a list of `NestedMap`, one for each split.

Returns a 3-tuple with:

*   weighted scalars, a NestedMap containing str keys and (value, weight)
    pairs for the current batch (a tuple of two scalars).
*   results, a `.NestedMap` as decoder output.
*   metrics, a NestedMap containing str keys and clu_metrics.Metric
    objects.

#### `process_decode_out`

*Processes one batch of decoded outputs.*

This code will run on the host (CPU) and not on an accelerator (GPU or
TPU). This allows you to run things that can't be processed on TPUs, such as
strings.

Args:

*   input_obj: The input object where a tokenizer is accessible.
*   decode_out: The output from decode(). May have an extra leading axis.

Returns a 3-tuple with:

*   weighted scalars, a NestedMap containing str keys and (value, weight)
    pairs for the current batch (a tuple of two scalars).
*   A list of tuples where each element corresponds to a row in the batch.
    Each tuple is a key value pair.
*   metrics, a NestedMap containing str keys and clu_metrics.Metric
    objects. These will run outside of pmap/pjit.

---

### Select an Existing Model

TODO: To be written. Volunteers welcome.

For a library of pre-defined models, check out [base models][base-models], which
includes:

*   LanguageModel
*   SequenceModel
*   ClassificationModel


<!-- Reference Links -->

[base-models]: https://github.com/google/praxis/tree/main/praxis/layers/models.py
[experiments]: https://github.com/google/paxml/tree/main/paxml/docs/experiments.md
[model_ipynb]: https://github.com/google/paxml/tree/main/paxml/docs/hands-on-tutorials.md#pax-model-and-task
[pax-elements]: https://github.com/google/paxml/tree/main/paxml/docs/learning-pax.md#pax-elements
[tasks]: https://github.com/google/paxml/tree/main/paxml/docs/tasks.md
