FROM python:3.5

ENV LANG C.UTF-8  
ENV LC_ALL C.UTF-8

RUN pip install tensorflow==1.8.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    && pip install mxnet -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    && pip install annoy scipy sklearn easydict opencv-python scikit-image flask gunicorn requests  -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    && mkdir /tmp/search
	
WORKDIR /opt/insightface/src/api
CMD ["/bin/bash","-c","gunicorn -w $WORKERS --threads $THREADS -b 0.0.0.0:$PORT api_flask:app"]

