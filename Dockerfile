FROM ubuntu

WORKDIR /app
RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip install --upgrade pip
RUN pip install matplotlib numpy cvxpy
RUN mkdir graphs


COPY . .

CMD [ "python3", "experiments.py"] 

