FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /project

ARG CXX=g++
ARG CXXFLAGS="--std=c++11"

# Install essential packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    curl \
    g++

COPY dev-requirements.txt requirements.txt /project

# Install python packages
RUN pip3 install -r dev-requirements.txt

RUN rm -rf dev-requirements.txt requirements.txt

CMD ["bash"]
