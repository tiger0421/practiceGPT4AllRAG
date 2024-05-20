FROM python:3.11.9-slim

WORKDIR /root
COPY ./dl_model.py /tmp

RUN apt update && \
    apt upgrade -y && \
    apt install -y --no-install-recommends \
        build-essential \
        cmake \
        locales \
        ffmpeg \
        libsm6 \
        libxext6 \
        poppler-utils \
        tesseract-ocr \
    && \
    pip install --no-cache-dir \
        langchain==0.1.20 \
        langchain_community==0.0.38 \
        transformers==4.40.2 \
        sentence-transformers==2.7.0 \
        faiss-cpu==1.8.0 \
        unstructured==0.13.7 \
        unstructured[all-docs] \
    && \
    apt --purge remove -y build-essential && \
    apt clean  && \
    apt autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8

WORKDIR /root/practiceGPT4AllRAG
RUN python /tmp/dl_model.py && \
    python3 -m nltk.downloader all -d /root/nltk_data

COPY ./test.pdf /tmp
COPY ./save.py /tmp
RUN python /tmp/save.py && \
    rm /tmp/*

ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
