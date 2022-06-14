ARG cpu_base_image="ubuntu:20.04"
ARG base_image=$cpu_base_image
FROM $base_image

LABEL maintainer="Pax team <pax-dev@google.com>"

# Re-declare args because the args declared before FROM can't be used in any
# instruction after a FROM.
ARG cpu_base_image="ubuntu:20.04"
ARG base_image=$cpu_base_image

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

RUN git clone https://github.com/google/paxml.git
RUN git clone https://github.com/google/praxis.git

RUN mv praxis/praxis /paxml/

RUN pip3 install paxml/paxml/pip_package
RUN pip3 install paxml/praxis/pip_package

RUN git clone https://github.com/google/flaxformer.git
RUN cd flaxformer && pip3 install .

RUN cd paxml && bazel build paxml/tasks/lm/params:main

RUN pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
RUN pip3 install fiddle-config tensorstore
RUN pip3 uninstall -y seqio seqio-nightly
# Need tensor2tensor explicitly because seqio implicitly dep on tensor2tensor.
RUN pip3 install -U seqio-nightly tensor2tensor

WORKDIR "/paxml"

CMD ["/bin/bash"]
