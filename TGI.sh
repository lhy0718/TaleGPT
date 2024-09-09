# Setting up the environment for the TGI pipeline (Inference Endpoint)

model=LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct

host_cache=$PWD/../.cache/huggingface/
host_tokenizer_config_path=$PWD/tokenizer_config.json

container_cache=/root/.cache/huggingface/
container_hub=/root/.cache/huggingface/hub
container_tokenizer_config_path=/tokenizer-config.json

# options: https://huggingface.co/docs/text-generation-inference/main/en/architecture#routers-command-line
docker run -it --rm --gpus all --shm-size 1g -p 8080:80 \
    -v $host_cache:$container_cache \
    -v $host_tokenizer_config_path:$container_tokenizer_config_path \
    ghcr.io/huggingface/text-generation-inference:2.2.0 \
    --model-id $model \
    --dtype bfloat16 \
    --trust-remote-code \
    --max-concurrent-requests 2 \
    --huggingface-hub-cache $container_hub \
    --tokenizer-config-path $container_tokenizer_config_path