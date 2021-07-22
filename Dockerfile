ARG BASE_IMAGE=sinzlab/pytorch:v3.8-torch1.7.0-cuda11.0-dj0.12.7
ARG REPOS='neuralpredictors nnfabrik mei vivid'

# Perform multistage build to pull private repo without leaving behind
# private information (e.g. SSH key, Git token)
FROM ${BASE_IMAGE} as base
ARG DEV_SOURCE=sinzlab
ARG UPSTREAM=sinzlab
ARG REPOS

# GitHub username and GitHub Personal Access Token must be specified
ARG GITHUB_USER
ARG GITHUB_TOKEN

WORKDIR /src
# Use git credential-store to specify username and pass to use for pulling repo
RUN git config --global credential.helper store &&\
    echo https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com >> ~/.git-credentials


RUN for REPO in $REPOS;\
    do \
    cd /src;\
    git clone https://github.com/${DEV_SOURCE}/${REPO};\
    cd /src/${REPO};\
    git remote add upstream https://github.com/${UPSTREAM}/${REPO};\
    done


RUN cd /src &&\
    git clone -b v1.7.1 https://github.com/cajal/dynamic-vision &&\
    git clone -b v2 https://github.com/cajal/utils


# Building the second stage
FROM ${BASE_IMAGE}
ARG REPOS
# copy everything found in /src over
# and then install them
COPY --from=base /src /src

RUN for REPO in ${REPOS};\
    do \
    pip install -e /src/${REPO};\
    done

RUN pip install -e /src/utils &&\
    pip install -e /src/dynamic-vision

RUN pip install streamlit paramiko slackclient boto3

# copy this project and install
COPY . /src/nnexplore
RUN pip install -e /src/nnexplore