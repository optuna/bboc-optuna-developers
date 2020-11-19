FROM python:3.6

RUN pip install --upgrade pip

ADD ./environment.txt /usr/src/environment.txt
RUN pip install --no-cache-dir -r /usr/src/environment.txt

# Installs google cloud sdk, this is mostly for using gsutil to export model.
# See https://cloud.google.com/ai-platform/training/docs/custom-containers-training
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    rm -rf /root/tools/google-cloud-sdk/.install/.backup
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

# Install dependencies for slack notification
RUN pip install --no-cache-dir requests

ADD ./submissions/mksturbo/requirements.txt /usr/src/requirements-mksturbo.txt
RUN pip install --no-cache-dir -U -r /usr/src/requirements-mksturbo.txt

ADD ./input /usr/src/input
ADD ./run_local.sh /usr/src/run_local.sh
ADD ./run_benchmark.py /usr/src/run_benchmark.py
ADD ./submissions /usr/src/submissions
WORKDIR /usr/src

CMD ["python", "run_benchmark.py"]
