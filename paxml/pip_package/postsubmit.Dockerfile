ARG image_name
ARG base_image="gcr.io/pax-on-cloud-project/${image_name}:latest"
FROM $base_image

RUN rm -rf /praxis && rm -rf /paxml/paxml && rm -rf /paxml/praxis
COPY . /paxml_new
RUN git clone https://github.com/google/praxis.git
RUN mv /praxis/praxis /paxml/ && mv /paxml_new/paxml /paxml/
RUN pip3 uninstall -y fiddle
RUN pip3 uninstall -y seqio
RUN pip3 install -r /paxml/paxml/pip_package/requirements.txt

RUN cd /paxml && bazel build ...
RUN cd /paxml && \
  bazel test \
  --test_output=all \
  --test_verbose_timeout_warnings \
  -- \
  paxml/... \
  -paxml/tasks/vision:input_generator_test

WORKDIR /

CMD ["/bin/bash"]
