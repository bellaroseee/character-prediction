FROM nvidia/cuda:10.2-devel
RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# You should install any dependencies you need here.
# RUN pip install tqdm
RUN pip install pandas
RUN pip install tensorflow
RUN pip install matplotlib
RUN pip install langdetect
