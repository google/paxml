ARG cpu_base_image="ubuntu:20.04"
ARG base_image=$cpu_base_image
FROM $base_image

LABEL maintainer="Pax team <pax-dev@google.com>"

# Re-declare args because the args declared before FROM can't be used in any
# instruction after a FROM.
ARG cpu_base_image="ubuntu:20.04"
ARG base_image=$cpu_base_image

ENV PYTHON_VERSION="3"
ENV PYTHON_MINOR_VERSION="8"

# Pick up some TF dependencies
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends software-properties-common
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
        aria2 \
        build-essential \
        curl \
        dirmngr \
        emacs \
        git \
        gpg-agent \
        less \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng-dev \
        libzmq3-dev \
        lsof \
        pkg-config \
        rename \
        rsync \
        python-dev \
        python3-distutils \
        sox \
        unzip \
        vim \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install python 3.8
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.8 python3.8-dev python3-pip python3.8-venv && \
    rm -rf /var/lib/apt/lists/* && \
    python3.8 -m pip install pip --upgrade && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 0

# Make python3.8 the default python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 0


ARG bazel_version=5.1.1
# This is to install bazel, for development purposes.
ENV BAZEL_VERSION ${bazel_version}
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

COPY . /paxml
RUN mkdir /tmp/paxml_pip_package_build/ && \
  cd paxml && git log | head -n 1 | cut -d' ' -f2 \
  > /tmp/paxml_pip_package_build/paxml_commit.txt

RUN git clone https://github.com/google/praxis.git && \
  cd praxis && git log | head -n 1 | cut -d' ' -f2 \
  > /tmp/paxml_pip_package_build/praxis_commit.txt

RUN cp -r praxis/praxis /paxml/

RUN pip3 install paxml/praxis/pip_package
RUN pip3 install paxml/paxml/pip_package

RUN git clone https://github.com/google/flaxformer.git
RUN cd flaxformer && pip3 install .

RUN mv paxml/paxml/pip_package /paxml/
RUN cd /paxml && bash pip_package/build.sh
#TODO:enable -praxis/layers:normalizations_test once the new Lingvo pip package is released
RUN cd praxis && bazel test --test_output=all --test_verbose_timeout_warnings -- praxis/... -praxis/layers:transformer_models_test -praxis/layers:flaxformer_models_test -praxis/layers:ngrammer_test -praxis/layers:attentions_test -praxis/layers:transformers_test -praxis/layers:models_test -praxis/layers:convolutions_test -praxis/layers:embedding_softmax_test -praxis/layers:ctc_objectives_test -praxis/layers:normalizations_test
RUN cd praxis && bash praxis/pip_package/build_pip_pkg.sh

WORKDIR /

CMD ["/bin/bash"]
