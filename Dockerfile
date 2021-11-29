# syntax=docker/dockerfile:1
FROM python:3
FROM ubuntu
ENV PYTHONUNBUFFERED=1
RUN apt-get update
RUN apt-get install -y git
WORKDIR /code
RUN git clone https://github.com/CrowStream/recommendation-ms.git
COPY requirements.txt /code/
RUN pip install -r requirements.txt
COPY . /code/
CMD python /recommendation/manage.py migrate