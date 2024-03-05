FROM python:3.12.2-bullseye

ARG DEBIAN_FRONTEND=noninteractive

COPY requirements.txt /
RUN python3 -m pip install -r /requirements.txt

COPY ./ttast /usr/lib/python3/dist-packages

ARG BUILD_VERSION
RUN echo "${BUILD_VERSION}" > /build_version.txt

ENTRYPOINT [ "/entrypoint" ]
