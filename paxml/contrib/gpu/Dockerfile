ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:23.01-tf2-py3
FROM ${FROM_IMAGE_NAME}

RUN pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install git+https://github.com/google-research/t5x

WORKDIR /pax
ENV PYTHONPATH=/pax:/pax/paxml:/pax/praxis

RUN pip install --force-reinstall git+https://github.com/google/fiddle
RUN git clone https://github.com/google/praxis \
  && cd praxis/praxis/pip_package \
  && pip install .

COPY . /pax/paxml

RUN pip install -e paxml/paxml/pip_package[gpu]

WORKDIR /pax/paxml
