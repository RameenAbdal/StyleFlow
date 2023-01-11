FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu18.04

RUN apt-get update && apt-get install python3.7 python3-dev libpython3.7-dev python3-pip cmake g++ gnupg -y
RUN apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 libsm6 libxext6 libxrender-dev x11-apps libqt5x11extras5 -y

ENV DEBIAN_FRONTEND noninteractive
RUN apt install locales -y
RUN sed -i -e 's/# en_US.UTF-8/en_US.UTF-8/' /etc/locale.gen && locale-gen
ENV LANG=en_US.UTF-8 \
        LANGUAGE=en_US:en \
        LC_ALL=en_US.UTF-8

RUN python3.7 -m pip install --upgrade pip setuptools

WORKDIR /usr/app
RUN python3.7 -m pip install torch==1.8.2 torchvision==0.9.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
COPY ./requirements.txt /usr/app
RUN --mount=type=cache,target=/root/.cache \
        python3.7 -m pip install -r requirements.txt

COPY . /usr/app
