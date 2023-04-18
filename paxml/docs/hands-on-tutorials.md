# Hands-on Tutorials

doc/pax/hands-on-tutorials

[TOC]

TLDR: A curated set of Jupyter Notebooks to get you comfortable using Pax.

## Overview

The following Jupyter Notebooks have been create by Pax SMEs. The goal is to provide a hands-on introduction
basic Pax activities.

Good luck! Have fun!

## Jupyter Notebooks

### A JAX Primer

In this first Jupyter Notebook, you will be introduced to JAX basics. When finished, you
will be ready to dive in and see how JAX is used in the Pax infrastructure for
training and multipod models.

**Jupyter Notebook (coming soon):** [A JAX Primer][ipynb_jax_primer]

### JAX for Edit-Distance

This lab showcases a few new things you can do in Jupyter Notebook with JAX. Specifically,
you will develop a native, JAX-based edit-distance algorithm that works on
padded batches.

You are encouraged to build your own JAX Jupyter Notebooks as a way to learn by doing.

**Jupyter Notebook (coming soon):** [JAX for Edit-Distance][ipynb_jax_ed]

### Pax Layer Basics

Get your first hands-on look at Pax. You will learn about its fundamental
component, the Pax Layers: the essential building blocks of models. In the
process, you will learn about the basics for authoring a new Pax layer.

Additionally, you will learn about Flax, a high-performance neural network
library for JAX that is designed for flexibility.

**Jupyter Notebook:** [Pax Layer Basics][ipynb_pax_layer]

### Inputs in Pax

This short Jupyter Notebook demonstrates how inputs work in Pax.

**Jupyter Notebook:** 
* [Inputs in Pax (training)][ipynb_pax_inputs_train]
* [Inputs in Pax (eval)][ipynb_pax_inputs_eval]

### Pax Model and Task

Model and task examples in Pax

**Jupyter Notebook (coming soon):** [Pax Model and Task][ipynb_model_and_task]

### Pax End-to-end Tutorial

Here you will put together what you have learned so far. You will be building a
Translator using a Transformer Encoder/Decoder Architecture via the Pax
framework.

Without going too deep into the Layer design, you will see how to build, test,
and combine them to train a model.

**Jupyter Notebook:** [Pax End-to-end Tutorial][ipynb_pax_e2e]

### Pax Checkpointing

This lab shows how to use checkpoints for warm-starting. It covers three topics
around checkpoint handling:

*   **Run a fine-tuning** starting from a pretrained checkpoint in Pax.
*   **Inspect variables** in a checkpoint file.
*   **Test a model** interactively using a checkpoint file.

**Jupyter Notebook (coming soon):** [Pax Checkpointing][ipynb_checkpoint]

### Pax RNN Decode

This Jupyter Notebook demonstates how to set up *extend step*, *init*, and *update decode
state cache* for the model's autoregressive decoding.

In *autoregressive decoding*, each output of the network is generated based on
previously generated output. Models such as RNN and transformer uses
autoregressive decoding to generate new sequence.

**Jupyter Notebook (coming soon):** [Pax RNN Decode][ipynb_rnn_decode]

### Sharding in Pax

This Jupyter Notebook demonstrates how to use *sharding* in Pax.

**Jupyter Notebook (coming soon):** [Sharding Jupyter Notebook][ipynb_shard_hard]


## Where to go from here

Congratulations! At this point, you should have a pretty good idea on how to
work with the various aspects of Pax. The rest of this site should help guide
you as you move on to deeper topics.


<!-- Reference Links -->

[ipynb_shard_hard]: https://github.com/google/paxml/tree/main/paxml/docs/tutorials/sharding.ipynb
[ipynb_checkpoint]: https://github.com/google/paxml/tree/main/paxml/docs/tutorials/pax201_checkpointing.ipynb
[ipynb_jax_ed]: https://github.com/google/paxml/tree/main/paxml/docs/tutorials/pax101_jax_for_edit_distance.ipynb
[ipynb_jax_primer]: https://github.com/google/paxml/tree/main/paxml/docs/tutorials/pax101_jax_primer.ipynb
[ipynb_model_and_task]: https://github.com/google/paxml/tree/main/paxml/docs/tutorials/pax101_model_and_task.ipynb
[ipynb_pax_e2e]: https://github.com/google/paxml/tree/main/paxml/docs/tutorials/pax101_e2e_tutorial.ipynb
[ipynb_pax_inputs_train]: https://github.com/google/paxml/blob/main/paxml/docs/tutorials/inputs_in_Pax-train.ipynb
[ipynb_pax_inputs_eval]: https://github.com/google/paxml/blob/main/paxml/docs/tutorials/inputs_in_Pax-eval.ipynb
[ipynb_pax_layer]: https://github.com/google/paxml/tree/main/paxml/docs/tutorials/pax_layer_basics.ipynb
[ipynb_rnn_decode]: https://github.com/google/paxml/tree/main/paxml/docs/tutorials/pax_rnn_decode.ipynb
