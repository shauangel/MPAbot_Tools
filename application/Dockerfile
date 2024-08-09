# syntax=docker/dockerfile:1.4
# Start from the Python image, automatically retrieved from Docker Hub
FROM --platform=$BUILDPLATFORM python:3.9-slim-bullseye AS builder

# work dir application's code
WORKDIR /src
COPY requirements.txt /src
COPY /Users/shauangel/.cache/kagglehub/models /kaggle
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt
RUN pip3 list
# Install our dependencies, by running a command in the container
# RUN pip install --upgrade pip --no-cache-dir
# RUN pip install -r /application/requirements.txt --no-cache-dir
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download en_core_web_md

# RUN /usr/src/application/lang_mod.sh

COPY . .

ENV PYTHONPATH "${PYTHONPATH}:/models"

CMD ["python3", "toolbox.py"]

FROM builder as dev-envs

RUN <<EOF
apk update
apk add git
EOF

RUN <<EOF
addgroup -S docker
adduser -S --shell /bin/bash --ingroup docker vscode
EOF

# install Docker tools (cli, buildx, compose)
COPY --from=gloursdocker/docker / /

CMD ["python3", "toolbox.py"]