# syntax=docker/dockerfile:1
FROM python:3
FROM ubuntu
ENV PYTHONUNBUFFERED=1
RUN apt-get update
RUN apt-get install -y git
RUN apt install -y python3-pip
RUN git clone https://github.com/CrowStream/recommendation-ms.git
WORKDIR /recommendation-ms/
RUN pip install -r requirements.txtls
