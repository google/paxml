# Pax Key Concepts

doc/pax/concepts

[TOC]

The following are *additional* terms and concepts that you are likely to
encounter when working with Pax. (This section assumes you are already familiar
with the concepts introduced on the [Learning Pax][learning-pax] page.)

## AutoML/Tuning

**AutoML** is the automated process of tuning machine learning tasks. This can
include hyperparameter tuning as well as **Neural Architecture Search** (NAS)—a
technique for automating the design of neural networks.

Pax uses [PyGlove][pyglove] for its hyperparameter tuning and AutoML.

For specific Pax-related AutoML tasks, see the How-to article:
[AutoML/Tuning (PyGlove)][automl].

## Checkpoints (Orbax)

A **checkpoint** is a saved snapshot of a model's internal state, including
weights, biases, learning rate, etc. If you want to restart the training from a
specific time in the training (such as trying to recover from a system failure)
the saved checkpoint can be used to resume from where it left off.

Checkpointing is also useful when you want to **fine-tune** a model: starting a
new model from a *different* pre-trained model.

Pax uses Orbax to provide a flexible and customizable approach for managing
checkpoints for various different objects.

Jupyter Notebook resources:

*   [Pax Checkpointing][checkpoint-colab] - A hands-on tutorial for Pax checkpointing.
*   [Checkpointing with Orbax][orbax] - Orbax example.
*   [Checkpoints (Orbax)][checkpoints] - Details specific Pax-related
    checkpointing tasks.

## Decode a Model

**Decoding** is one of the two key processes associated with *Transformer*
models. (The other is **encoding**.) Decoding is most often used for predicting,
translating, or some form of generating text.

In the context of Pax, decoding can be thought of as inference; though how it
will specifically be used will depend on the model.

When working with Pax, you will typically either be running a forward
propagation (`__call__()`) or a `decode()` method.

For specific Pax-related decode tasks, see the How-to article: [Decode][decode].

## Evaluation

Machine learning model **evaluation** metrics are used to:

*   Assess the quality of the fit between the model and the data
*   Compare models
*   Predict how accurate each model can be expected to perform on a specific
    data set

For supervised learning models, an evaluation set, the data not seen by the
model, is typically excluded from the training set to evaluate model
performance.

For unsupervised learning models, model evaluation is less defined and often
varies from model to model. Because unsupervised learning models do not reserve
an evaluation set, the evaluation metrics are calculated using the whole input
data set.

For more, see [BigQuery ML Model Evaluation Overview][eval].

## jax.Array
**jax.Array** is a unified array object that helps make parallelism a core
feature of JAX, simplifies and unifies JAX internals, and allows us to unify jit
and pjit.

*   [Jax.Array migration][jax-array-migration]
*   [Distributed arrays and automatic parallelization][jax-array]
*   [Using JAX in multi-host and multi-process environments][jax-multi-host]

## Metrics

Metrics is the general term for the process of generating values that determine
how "successful" your model is. Metrics can be used when training or evaluating
your model, and can include: Accuracy, Precision, Recall, F1 Score, Log
Loss/Binary Crossentropy, AUC, etc.

Pax supports two different metric types:

*   **WeightedScalars**: If your metric can be computed via a weighted mean
    across samples, WeightedScalars is a simple approach.
*   **`clu.metrics`**: a metrics library—used by several other Flax/Jax
    projects—which are much more flexible, allowing you to store arbitrary
    JTensors on the metric object, and perform arbitrary metric computations.

For specific Pax-related metrics tasks, see the How-to article:
[Metrics][metrics].

## TensorBoard

**TensorBoard** is a managed service that enables hosting, tracking, and sharing
of ML *experiments*. As a result, it makes collaboration easier in ML research.

Pax uses TensorBoard for visualizing and understanding ML runs and models.

Check out [TensorBoard][tensorboard] for more.


<!-- Reference Links -->

[automl]: https://github.com/google/paxml/tree/main/paxml/docs/automl.md
[checkpoint-colab]: https://github.com/google/paxml/tree/main/paxml/docs/tutorials/pax201_checkpointing.ipynb
[checkpoints]: https://github.com/google/paxml/tree/main/paxml/docs/model_checkpoint.md
[orbax]: https://github.com/google/orbax/blob/main/docs/checkpoint.md
[decode]: https://github.com/google/paxml/tree/main/paxml/docs/decoder.md
[eval]: https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-evaluate-overview
[jax-multi-host]: https://jax.readthedocs.io/en/latest/multi_process.html#local-vs-global-devices
[learning-pax]: https://github.com/google/paxml/tree/main/paxml/docs/learning-pax.md
[jax-array-migration]: https://jax.readthedocs.io/en/latest/jax_array_migration.html
[jax-array]: https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html
[metrics]: https://github.com/google/paxml/tree/main/paxml/docs/metrics.md
[pyglove]: internal-link/pyglove
[tensorboard]: internal-link/tensorboard
[jax-array-migration]: https://jax.readthedocs.io/en/latest/jax_array_migration.html
[jax-array]: https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html
