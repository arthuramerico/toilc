#!/bin/bash
docker build -t toilc .
docker run  --name experiments toilc
docker cp experiments:/app/graphs .
docker container rm experiments

