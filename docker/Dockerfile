FROM python:3.12.3-slim

WORKDIR /root

RUN apt update && \
    apt upgrade -y && \
    apt install -y --no-install-recommends \
        build-essential \
        locales \
        git \
    && \
    pip install --no-cache-dir \
        langchain \
        langchain_community \
        transformers \
        faiss-cpu \
        unstructured \
    && \
    git clone https://github.com/tiger0421/practiceGPT4AllRAG.git && \
    python practiceGPT4ALLRAG/docker/faiss_test.py && \
    apt --purge remove -y build-essential git && \
    apt clean  && \
    apt autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8

ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9

