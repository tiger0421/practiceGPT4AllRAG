from huggingface_hub import snapshot_download

download_path = snapshot_download(repo_id="intfloat/multilingual-e5-large", local_dir="/root/practiceGPT4AllRAG/models/intfloat_multilingual-e5-large")
