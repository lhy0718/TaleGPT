# Chat UI
# https://huggingface.co/docs/chat-ui/index

docker run -p 3000:3000 --env-file .env.local -v chat-ui:/data --name chat-ui ghcr.io/huggingface/chat-ui-db
