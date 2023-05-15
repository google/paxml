# coding=utf-8
# Copyright 2022 The Pax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""tf.data service library.

This library is designed based on the idea of using the same binary
for all jobs (client, tf.data service dispatcher, and tf.data service
workers). It defines all relevant flags in one place.

The API is just two functions:

1. In the main function, the users should add something like:
```
  if tf_data_service_lib.run_tf_data_service(FLAGS.mode):
    return
```

2. and the client (trainer) code can access the tf.data service
address using `get_tf_data_service_address()`.
"""
from absl import flags
from absl import logging
import tensorflow as tf
from typing import Optional

# TODO(b/281571038): When this feature is available in
# xm_tf_data_service.py, update pax/xm_launch.py to enable the prefix.

# Flags for both tf.data service dispatcher and workers.
_PORT = flags.DEFINE_integer(
    'tf_data_service_port',
    0,
    (
        'Specifies the port to bind to. A value of 0 indicates that the server'
        ' may bind to any available port.'
    ),
)

# Flags for tf.data service workers.
_DISPATCHER_ADDRESS = flags.DEFINE_string(
    'tf_data_service_dispatcher_address',
    None,
    'Specifies the address of the dispatcher.',
)
_WORKER_ADDRESS = flags.DEFINE_string(
    'tf_data_service_worker_address',
    None,
    (
        'Specifies the address of the worker server. This address is passed to'
        ' the dispatcher so that the dispatcher can tell clients how to connect'
        ' to this worker.'
    ),
)

_DATA_TRANSFER_PROTOCOL = flags.DEFINE_string(
    'tf_data_service_data_transfer_protocol',
    None,
    'A string indicating the protocol to be used by the worker to transfer data'
    ' to the client. E.g. "grpc".',
)

_DATA_TRANSFER_ADDRESS = flags.DEFINE_string(
    'tf_data_service_data_transfer_address',
    None,
    'A string indicating the data transfer address of the worker server.',
)

# Flags for tf.data service dispatcher.
_WORK_DIR = flags.DEFINE_string(
    'tf_data_service_work_dir',
    None,
    (
        'A directory to store dispatcher state in. This argument is required'
        ' for the dispatcher to be able to recover from restarts.'
    ),
)
_FAULT_TOLERANT_MODE = flags.DEFINE_bool(
    'tf_data_service_fault_tolerant_mode',
    False,
    (
        'Whether the dispatcher should write its state to a journal so that it'
        ' can recover from restarts. Dispatcher state, including registered'
        ' datasets and created jobs, is synchronously written to the journal'
        ' before responding to RPCs. If `True`, `work_dir` must also be'
        ' specified.'
    ),
)

_WORKER_ADDRESSES = flags.DEFINE_list(
    'tf_data_service_worker_addresses',
    None,
    '(Optional.) If the job uses static sharding with fixed replicas, it needs'
    ' to specify a list of comma-separated worker addresses that will register'
    ' with the dispatcher. The worker addresses should be in the format "host"'
    ' or "host:port", where "port" is an integer, named port, or %port% to'
    ' match any port.',
)

# Flags for tf.data service client.
_TF_DATA_SERVICE_ADDRESS = flags.DEFINE_string(
    'tf_data_service_address',
    None,
    'The address of the tf.data service.',
)


def _get_worker_config() -> tf.data.experimental.service.WorkerConfig:
  return tf.data.experimental.service.WorkerConfig(
      dispatcher_address=_DISPATCHER_ADDRESS.value,
      worker_address=_WORKER_ADDRESS.value,
      port=_PORT.value,
      data_transfer_protocol=_DATA_TRANSFER_PROTOCOL.value,
      data_transfer_address=_DATA_TRANSFER_ADDRESS.value,
  )


def _get_dispatcher_config() -> tf.data.experimental.service.DispatcherConfig:
  return tf.data.experimental.service.DispatcherConfig(
      port=_PORT.value,
      work_dir=_WORK_DIR.value,
      fault_tolerant_mode=_FAULT_TOLERANT_MODE.value,
      worker_addresses=_WORKER_ADDRESSES.value,
  )


def get_tf_data_service_address() -> Optional[str]:
  return _TF_DATA_SERVICE_ADDRESS.value


def run_tf_data_service(mode: str) -> bool:
  """Maybe run tf.data service servers.

  If running in 'tf_data_service_dispatcher' or
  'tf_data_service_worker' mode, this function runs those servers, and
  blocks until the server has shut down, and returns True to the
  caller.

  Otherwise returns False immediately.

  Args:
    mode: if this is equal to either 'tf_data_service_dispatcher' or
      'tf_data_service_worker', runs the corresponding tf.data service server.

  Returns:
    True if a tf.data service server was run. (Callers would probably
    want to exit from main in this case.)
  """
  if mode == 'tf_data_service_dispatcher':
    logging.info('run_tf_data_service(mode=%s)', mode)
    dispatcher = tf.data.experimental.service.DispatchServer(
        config=_get_dispatcher_config()
    )
    dispatcher.join()
    logging.warning('Done.')
    return True

  if mode == 'tf_data_service_worker':
    logging.info('run_tf_data_service(mode=%s)', mode)
    worker = tf.data.experimental.service.WorkerServer(
        config=_get_worker_config()
    )
    worker.join()
    logging.warning('Done.')
    return True

  return False
