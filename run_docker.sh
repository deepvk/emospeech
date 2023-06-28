#!/bin/bash

app=$PWD

docker run --name emospeech -it --rm \
    --net=host --ipc=host \
    --gpus "all" \
    -v "$app":/app \
    emospeech
