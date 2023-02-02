# Version: 0.2.2
## Major Features and Improvements
## Breaking changes
## Deprecations
## Note
*   Version: 0.2.2
*   Build Date: 20230201
*   Paxml commit: cab3ed811174682733e2c836363510162fbfb1da
*   Praxis version: 0.2.2
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
