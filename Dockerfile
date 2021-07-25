FROM sinzlab/pytorch:v3.8-torch1.7.0-cuda11.0-dj0.12.7

#RUN cd /src; \
#    git clone -b v0.0 https://github.com/sinzlab/neuralpredictors; \
#    pip3 install -e /src/neuralpredictors

# copy this project and install
COPY . /src/LVIV-2021
RUN pip install -e /src/LVIV-2021

WORKDIR /content

