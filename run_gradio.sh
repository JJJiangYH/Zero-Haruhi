export CUDA_VISIBLE_DEVICES=0
export HF_HOME="/workspace/jyh/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"

# Start the gradio server
/workspace/jyh/miniconda3/envs/ChatWorld/bin/python /workspace/jyh/Zero-Haruhi/app.py