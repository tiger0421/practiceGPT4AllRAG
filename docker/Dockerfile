FROM python:3.11.9-slim

WORKDIR /root
COPY ./dl_model.py /tmp

RUN apt update && \
    apt upgrade -y && \
    apt install -y --no-install-recommends \
        build-essential \
        cmake \
        locales \
        git \
        ffmpeg \
        libsm6 \
        libxext6 \
    && \
    pip install --no-cache-dir \
        langchain \
        langchain_community \
        transformers \
        faiss-cpu \
        unstructured \
        unstructured[all-docs] \
        sentence-transformers \
    && \
    git clone https://github.com/tiger0421/practiceGPT4AllRAG.git && \
    python /tmp/dl_model.py && \
    apt --purge remove -y build-essential git && \
    apt clean  && \
    apt autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8

WORKDIR /root/practiceGPT4AllRAG

ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9

