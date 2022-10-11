ARG image_name
ARG base_image="gcr.io/pax-on-cloud-project/${image_name}:latest"
FROM $base_image

RUN rm -rf /praxis && rm -rf /paxml/paxml && rm -rf /paxml/praxis
COPY . /paxml_new
RUN git clone https://github.com/google/praxis.git
RUN mv /praxis/praxis /paxml/ && mv /paxml_new/paxml /paxml/
RUN pip3 install -U --no-deps -r /paxml/praxis/pip_package/requirements.txt
RUN pip3 install -U --no-deps -r /paxml/paxml/pip_package/requirements.txt
RUN cd /paxml && bazel build ...

RUN cd /paxml && bazel test paxml/... --test_output=all --test_verbose_timeout_warnings
WORKDIR /

CMD ["/bin/bash"]
