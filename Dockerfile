FROM python:3.10
WORKDIR /app

COPY . /app

VOLUME /app/data
RUN pip3 install -r requirements.txt

# RUN pip3 install tensorflow
RUN pip3 install tf-nightly
RUN pip3 install keras-nightly
# RUN pip3 install keras
RUN pip3 install pillow
# RUN python3 -c "from torchvision.models import mobilenet_v3_small; mobilenet_v3_small(pretrained=True)" 2321

RUN chmod +x /app/baseline.py
RUN chmod +x /app/make_submission.py

#CMD ["python3","/app/baseline.py"]
