ARG cpu_base_image="ubuntu:22.04"
ARG base_image=$cpu_base_image
FROM $base_image

LABEL maintainer="Pax team <pax-dev@google.com>"

# Re-declare args because the args declared before FROM can't be used in any
# instruction after a FROM.
ARG cpu_base_image="ubuntu:22.04"
ARG base_image=$cpu_base_image
ARG wheel_folder
ENV WHEEL_FOLDER $wheel_folder
ENV PYTHON_VERSION="3"
ENV PYTHON_MINOR_VERSION="10"

# Pick up some TF dependencies
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends software-properties-common
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        pkg-config \
        rename \
        rsync \
        unzip \
        vim \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install python 3.10
RUN apt-get update && apt-get install -y \
    python3 python3-dev python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/* && \
    python3.10 -m pip install pip --upgrade && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 0

# Install pip package
RUN pip3 install pip-tools

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

RUN mkdir -p $WHEEL_FOLDER && cd paxml && \
  git rev-parse HEAD > $WHEEL_FOLDER/paxml_commit.txt

RUN git clone https://github.com/google/praxis.git && \
  cd praxis && git rev-parse HEAD > $WHEEL_FOLDER/praxis_commit.txt

RUN cp -r praxis/praxis /paxml/

RUN cd /praxis && \
    pip-compile --quiet requirements.in \
    --output-file $WHEEL_FOLDER/praxis_requirements.txt

RUN cd /paxml && \
    sed -i 's/praxis/#praxis/' requirements.in && \
    pip-compile --quiet requirements.in \
    /praxis/requirements.in \
    --output-file $WHEEL_FOLDER/paxml_requirements.txt

RUN pip3 install --no-deps -r paxml/paxml/pip_package/requirements.txt

RUN mv paxml/paxml/pip_package /paxml/
RUN cd /paxml && bash pip_package/build.sh
#TODO:enable -praxis/layers:normalizations_test once the new Lingvo pip package is released
RUN cd praxis && \
  bazel test \
  --test_output=all \
  --test_verbose_timeout_warnings \
  -- \
  praxis/... \
  -praxis/layers:attentions_test \
  -praxis/layers:convolutions_test \
  -praxis/layers:ctc_objectives_test \
  -praxis/layers:embedding_softmax_test \
  -praxis/layers:models_test \
  -praxis/layers:ngrammer_test \
  -praxis/layers:normalizations_test \
  -praxis/layers:transformer_models_test \
  -praxis/layers:transformers_test

RUN cd praxis && bash praxis/pip_package/build_pip_pkg.sh

WORKDIR /

CMD ["/bin/bash"]
