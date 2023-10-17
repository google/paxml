# Version: 1.2.0
## Major Features and Improvements
## Breaking changes
## Deprecations
## Note
*   Version: 1.2.0
*   Build Date: 20231016
*   Paxml commit: 4486d690773ba5d15d9e54b05a4e916b1a955de9
*   Praxis version: 1.2.0
*   Praxis commit: 7bd63412bf86a68e09fcd9455f76a4909d19377e
# Version: 1.1.0
## Major Features and Improvements
* Support multislice on Cloud TPU: example configs and instructions added ([link](https://github.com/google/paxml#pax-on-multislice)).
* Move to python 3.10 as the minimal python requirement (previously on python 3.8).
* Checkpoint improvements:
  * OCDBT support (off by default).
  * Pax can take advantage of [Orbax-style transformations](https://github.com/google/orbax/blob/main/docs/checkpoint.md#transformations) to gain support for additional features.
  * Users can provide their own Checkpointer to allow for custom logic and reading from a custom checkpoint location.
* Eval customization support.
* Support for [tf.data.experimental.service](https://www.tensorflow.org/api_docs/python/tf/data/experimental/service).
## Note
*   Version: 1.1.0
*   Build Date: 20230824
*   Paxml commit: fe4d87fe90dcbe2ef6db881d495c7d40270a3033
*   Praxis version: 1.1.0
*   Praxis commit: 2a7d0407871502a1d79dcd0e01411e73f1d15d36
# Version: 1.0.0
## Major Features and Improvements
* **Fiddle** - Pax's layer library, Praxis, now has layers and BaseParameterizable configured with [Fiddle](https://github.com/google/fiddle), a Python-first configuration library. Fiddle reduces boilerplate, and adds productivity features including history tracking, graphviz visualization, support for aliasing objects, and more.
* **CLI Experiment and Data Injectability** - Pax users are now able to select which experiments to run without the need to recompile for each experiment. Using a CLI interface based on Fiddle, users can override subsets of the experiment’s canonical dataset.
* **CLU Metrics** - Through Praxis, Pax has adopted CLU metrics as its standard metric interface.  This allows other Jax/Flax codebases that have CLU metrics to use them in Praxis.
* **Orbax Integration** - Pax has consolidated on the [Orbax checkpoint library](https://github.com/google/orbax/tree/main/checkpoint) as the standard checkpointing library.  Orbax supports pjit, pmap, and Pathways checkpointing. Orbax facilitates checkpoint compatibility with other frameworks and provides common functionality throughout the JAX ecosystem.
* **Flax Interoperability** - Through Praxis, Pax now supports shape inference, `__call__` for forward propagation, and has adopted Linen’s AxisMetadata for its mid-level sharding APIs.  These changes improve interoperability with other Flax-based libraries such as T5X.
* **Custom Training Loop Support** - Pax now provides limited support for customized training pipelines. Users can define a custom "train program" to encapsulate their training logic and use it in BaseExperiment. Customizable training loops pave the path for features such as multi-task support.
* **SeqIO** - Pax has adopted [SeqIO](https://github.com/google/seqio). Supporting SeqIO allows users with existing SeqIO Tasks, input pipelines and evaluation workflows to use them directly in Pax. We are working to make the Pax-SeqIO evaluation setup more flexible and robust.
* **Documentation** - We have added [documentation](https://github.com/google/paxml/tree/main/paxml/docs) and Jupyter Notebook [tutorials](https://github.com/google/paxml/tree/main/paxml/docs/hands-on-tutorials.md). (Although you may notice some empty links in the doc, they are placeholders for upcoming docs.)
## Note
*   Version: 1.0.0
*   Build Date: 20230329
*   Paxml commit: 033eb2421a6fc3e24f76bb19dd260c6776c5933b
*   Praxis version: 1.0.0
*   Praxis commit: 621c2ca7bfcd0e21ea118a3d8e40e29b48313c0c
# Version: 0.4.0
## Note
*   Version: 0.4.0
*   Build Date: 20230329
*   Paxml commit: 033eb2421a6fc3e24f76bb19dd260c6776c5933b
*   Praxis version: 0.4.0
*   Praxis commit: 621c2ca7bfcd0e21ea118a3d8e40e29b48313c0c
# Version: 0.3.0
## Major Features and Improvements
* Partitioning API refactoring
* Orbax checkpointing integration
* Improve AutoML and hyperparameter search support
* Improve support for differential privacy
* Support for bool, int32, etc. as non-trainable parameters
## Note
*   Version: 0.3.0
*   Build Date: 20230201
*   Paxml commit: cab3ed811174682733e2c836363510162fbfb1da
*   Praxis version: 0.3.0
*   Praxis commit: 9e1d13d888ac18a567e249ddb41e6b1bd1fe505a
# Version: 0.2.1
## Bug fix
* fix the HBM OOM error when loading the checkpoint using orbax
## Note
*   Version: 0.2.1
*   Build Date: 20221121
*   Paxml commit: c4628a21946dd13eb5a42b6f5862284088d90730
*   Praxis version: 0.2.1
*   Praxis commit: f7e98026c1c5ecbc6e4aff175621d443fa37fcf2
# Version: 0.2.0
## Major Features and Improvements
* Revamp training and evaluation/decoding loops in preparation of multi-task
  learning support
* Better seqio metrics and clu metrics support
* Suppot per-core batch size < 1
* Introduction of training input specifications
* Add GPT-3-like Language Modeling configuration
* Add ResNet-50 image classification configuration
* Checkpointing
  - Save on-demand checkpoint when preemption signal is received
  - Migration to Orbax checkpointing
  - Save checkpoints more consistently after pre-emption and when training
    completes
* Summaries
  - Add support for aggregation across steps
* TPU profiling can be configured and automated
* Improvements to Auto-ML and hyperparameters sweeping using PyGlove
* Use etils for handling filenames / directories
## Note
*   Version: 0.2.0
*   Build Date: 20221114
*   Paxml commit: 16fd2143827522acfb7a7e22767a008eaae07e24
*   Praxis version: 0.2.0
*   Praxis commit: 413da1ad8148f27faebca119f8c5deedca66228b
# Version: 0.1.0
## Major Features and Improvements
## Breaking changes
## Deprecations
## Note
*   Version: 0.1.0
*   Build Date: 20220702
*   Commit: 546370f5323ef8b27d38ddc32445d7d3d1e4da9a
*   Praxis version: 0.1.0
