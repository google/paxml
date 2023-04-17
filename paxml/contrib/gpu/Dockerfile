ARG FROM_IMAGE_NAME=nvcr.io/nvidia/cuda:12.1.0-devel-ubuntu20.04
FROM ${FROM_IMAGE_NAME}

RUN apt update && apt install -y python3-pip &&  apt install -y git

RUN pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

WORKDIR /pax
ENV PYTHONPATH=/pax:/pax/paxml:/pax/praxis

RUN git clone https://github.com/google/praxis \
  && cd praxis \
  && git checkout 7b1d736fa96fd10c6205c323b2ea467741cd4638

RUN git clone https://github.com/google/paxml \
  && cd paxml \
  && git checkout 75d70d7994507974311b29254617d39c9cd4764e

COPY paxml/contrib/gpu/ paxml/paxml/contrib/gpu/
RUN pip install -r paxml/paxml/contrib/gpu/scripts_gpu/requirements.txt

ENV XLA_FLAGS="--xla_gpu_enable_triton_gemm=false ${XLA_FLAGS}"

WORKDIR /pax/paxml
