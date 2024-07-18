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

"""Expose functionalities for profiling code."""

from absl import logging
from ctypes import cdll

libcudart = cdll.LoadLibrary('libcudart.so')
def cudaProfilerStart():
    libcudart.cudaProfilerStart()
def cudaProfilerStop():
    libcudart.cudaProfilerStop()
def cudaDeviceSynchronize():
    libcudart.cudaDeviceSynchronize()

class Profiler:
  """Dummy class to capture code profiles.

  Note: The current implementation is a no-op.
  """

  def __init__(
      self,
      num_steps: float = 2.0,
      min_duration_sec: float = 1.0,
      default_duration_sec: float = 5.0,
      tag: str | None = None,
      max_num_hosts: int | None = None,
      capture_host_profile: bool = False,
  ) -> None:
    """Constructor.

    Args:
      num_steps: The number of steps to capture based on the step duration
        estimate that is set by calling update_step_moving_mean() successfully.
      min_duration_sec: The minimum duration of the profiler capture in seconds.
        Set to this value when the estimate step duration times num_steps is
        smaller than this value.
      default_duration_sec: The default duration of the profiler capture in
        seconds. Used when no step duration were sampled by calling
        update_step_moving_mean().
      tag: An optional tag to be added to the profiler trace.
      max_num_hosts: If max_num_hosts is unspecified, all hosts of devices will
        be chosen. Otherwise, at most max_num_hosts will be chosen. This option
        only works with pathways.
      capture_host_profile: Capture host CPU profile as well.
    """
    self._capture_num_steps = num_steps
    self._capture_min_duration_sec = min_duration_sec
    self._capture_default_duration_sec = default_duration_sec
    self._tag = tag
    self._max_num_hosts = max_num_hosts
    self._capture_host_profile = capture_host_profile
    self._step_duration_sec = 0.
    self._step_count = 0

  def capture_async(self) -> None:
    """Captures a trace asynchronously.

    The duration of the trace corresponds to step_duration_estimate_sec.
    """
    cudaProfilerStart()
    logging.info('Dummy profiler currently does not capture any trace.')

  def stop_capture_async(self) -> None:
    cudaDeviceSynchronize()
    cudaProfilerStop()

  def update_step_moving_mean(self, duration_sec: float):
    """Updates the step duration moving average with a step duration estimate.

    Args:
      duration_sec: The duration of the step to add in seconds.
    """
    self._step_duration_sec += duration_sec
    self._step_count += 1

  @property
  def step_duration_estimate_sec(self) -> float:
    """Estimates of the step duration in seconds.

    If update_step_moving_mean() has never been called, returns the default
    duration instead.
    """
    if not self._step_count:
      return self._capture_default_duration_sec
    return self._step_duration_sec / self._step_count
