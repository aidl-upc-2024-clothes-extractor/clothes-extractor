# set base image (host OS)
#FROM nvcr.io/nvidia/cuda:12.3.1-devel-ubuntu22.04
#FROM python:3.11.8-slim
FROM nvcr.io/nvidia/pytorch:24.01-py3


RUN apt-get update
RUN apt-get install ca-certificates -y
RUN apt-get install vim -y
RUN apt-get install curl -y 
RUN apt-get install nano -y
RUN apt-get install gdb -y
RUN apt-get clean

#~/.cache/torch/hub/checkpoints/ in MAC this is the cache of torch
#COPY data/cache_models /root/.cache/torch/hub/checkpoints

RUN apt update -y
RUN apt install -y libpq-dev gcc python3.10-venv
#python3.11-dev python3.11 

RUN rm -rf /var/cache/apt/archives || true
RUN rm -rf /var/lib/apt/lists/* || true

ENV VIRTUAL_ENV=/app
ENV CURLOPT_SSL_VERIFYHOST=0
ENV CURLOPT_SSL_VERIFYPEER=0

WORKDIR $VIRTUAL_ENV
RUN python -m venv $VIRTUAL_ENV

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python -m pip config set global.cert /usr/local/share/ca-certificates/firewall_root_base64.cer

# copy the dependencies file to the working directory
COPY requirements.txt .

# certificates
# RUN mkdir /usr/local/share/ca-certificates
COPY Certificates/firewall_root_base64.cer /usr/local/share/ca-certificates/firewall_root_base64.cer
RUN update-ca-certificates 2&>/dev/null || true

# install dependencies
#RUN python3.11 -m pip install -U torch --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host  pypi.python.org
RUN python -m pip install -r requirements.txt --trusted-host pypi.ngc.nvidia.com --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host  pypi.python.org

#~/.cache/torch/hub/checkpoints/ in MAC this is the cache of torch
RUN mkdir -p /root/.cache/torch/hub/checkpoints
COPY model_cache /root/.cache/torch/hub/checkpoints

# Copy code to the working directory
COPY . .

# # command to run on container start
# ENTRYPOINT [ "python", "entrypoint.py" ]

