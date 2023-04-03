ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:23.01-tf2-py3
FROM ${FROM_IMAGE_NAME}

WORKDIR /pax
ENV PYTHONPATH=/pax:/pax/paxml:/pax/praxis

RUN git clone https://github.com/google/praxis \
  && cd praxis \
  #&& git checkout 908aa066f02616acfdd94d26855f8cf8824665a9
  && git checkout 3aa307d635f53196c68371d77ac870f442c7abf8

## checkout commit for installing correct dependencies
RUN git clone https://github.com/google/paxml.git \
  && cd paxml \
  && git checkout a1245f2ded5a189d2e7b3f9a7b28f2f7f1fb3168

RUN pip install -r paxml/paxml/pip_package/requirements.txt
RUN pip install -r paxml/paxml/contrib/gpu/scripts_gpu/dataset_requirements.txt

## checkout working commit
RUN cd paxml && git checkout 644ca45173c58bd537374c911da375b70634263e

RUN pip install jax[cuda]==0.4.5 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

WORKDIR /pax/paxml
