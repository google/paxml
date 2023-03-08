ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:23.01-tf2-py3
FROM ${FROM_IMAGE_NAME}

RUN pip install jax[cuda]==0.4.3 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

WORKDIR /pax
ENV PYTHONPATH=/pax:/pax/paxml:/pax/praxis

RUN git clone https://github.com/google/praxis \
  && cd praxis/praxis/pip_package \
  && pip install .

COPY . /pax/paxml

RUN pip install -e paxml/paxml/pip_package[gpu]

WORKDIR /pax/paxml
