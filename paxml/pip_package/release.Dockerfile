ARG cpu_base_image="ubuntu:20.04"
ARG base_image=$cpu_base_image
FROM $base_image

LABEL maintainer="Pax team <pax-dev@google.com>"

# Re-declare args because the args declared before FROM can't be used in any
# instruction after a FROM.
ARG cpu_base_image="ubuntu:20.04"
ARG base_image=$cpu_base_image
ARG praxis_version
ARG wheel_folder
ENV WHEEL_FOLDER $wheel_folder
ENV PYTHON_VERSION="3"
ENV PYTHON_MINOR_VERSION="8"

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
RUN mkdir $WHEEL_FOLDER
RUN if [ "$praxis_version" = "release-test" ] ; then git clone https://github.com/google/praxis.git; else git clone -b r${praxis_version} https://github.com/google/praxis.git; fi
RUN if [ "$praxis_version" = "release-test" ] ; then sed -i 's/ @ git.*//g' /praxis/requirements.in; fi
RUN pip3 install -e praxis

RUN cp -r praxis/praxis /paxml/
RUN sed -i 's/ @ git.*//g' paxml/requirements.in
RUN pip3 install -r paxml/requirements.in

RUN mv paxml/paxml/pip_package /paxml/
RUN cd /paxml && bash pip_package/build.sh

WORKDIR /

CMD ["/bin/bash"]
