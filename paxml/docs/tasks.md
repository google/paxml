# Tasks

doc/pax/tasks

[TOC]

## Introduction

In general, an ML **task** identifies what you are trying to accomplish with
your ML model. Some examples of ML tasks include:

<section class="multicol">

<section>

*   [Regression][regression]
*   [Classification][classification]

</section>

<section>

*   [Clustering][clustering]
*   [Anomaly detection][anomaly]

</section>

<section>

*   [Transcription][transcription]
*   [Machine translation][translation]

</section>

</section>

In Pax, a **Task** is an object (derived from the [BaseTask][base-task] class)
that is composed of the necessary components to address a given ML task. These
components include:

*   A Model
*   Metrics (optional)
*   An optimizer

A **Mixture** is a term for *a collection of Tasks* and enables fine-tuning a
model on multiple Tasks simultaneously.

> Tip: For a rudamentary introduction to the basic Pax components, check out
> [Pax Elements][pax-elements].  If you want to dive in for a hands-on
> experience, try the [Pax Model and Task Jupyter Notebook][model_ipynb].

## Task How-To's

### Define a Task

A Task contains one or more Models and Learner/Optimizers. The simplest Task
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



<!-- Reference Links -->

[anomaly]: internal-link/ml-glossary/#anomaly-detection
[base-task]: https://github.com/google/paxml/tree/main/paxml/base_task.py
[classification]: internal-link/ml-glossary/#classification_model
[clustering]: internal-link/ml-glossary/#clustering
[model_ipynb]: https://github.com/google/paxml/tree/main/paxml/docs/hands-on-tutorials.md#pax-model-and-task
[pax-elements]: https://github.com/google/paxml/tree/main/paxml/docs/learning-pax.md#pax-elements
[regression]: internal-link/ml-glossary/#regression-model
[transcription]: https://fireflies.ai/blog/what-is-ai-transcription/
[translation]: https://en.wikipedia.org/wiki/Machine_translation
