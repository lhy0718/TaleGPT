# Setting up the environment for the TGI pipeline (Inference Endpoint)

model=LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct
# share a volume with the Docker container to avoid downloading weights every run
cache_host=$PWD/../.cache/huggingface/
tokenizer_config_path_host=$PWD/tokenizer_config.json

cache_container=/root/.cache/huggingface/
hub_container=/root/.cache/huggingface/hub
tokenizer_config_path_container=/tokenizer-config.json


docker run -it --rm --gpus all --shm-size 1g -p 8080:80 -v $cache_host:$cache_container -v $tokenizer_config_path_host:$tokenizer_config_path_container \
    ghcr.io/huggingface/text-generation-inference:2.2.0 \
    --model-id $model \
    --dtype bfloat16 \
    --trust-remote-code \
    --max-concurrent-requests 2 \
    --huggingface-hub-cache $hub_container \
    --tokenizer-config-path $tokenizer_config_path_container
