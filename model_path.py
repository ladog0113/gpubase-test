import os

from huggingface_hub import snapshot_download

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_ID = "google/gemma-4-31B-it"
DEFAULT_MODEL_DIR = os.path.join(BASE_DIR, "models", "gemma-4-31B-it")


def resolve(model):
    """로컬 경로면 그대로 쓰고, Hugging Face ID면 models/ 아래로 받아서 그 경로를 돌려준다."""
    if os.path.isfile(os.path.join(model, "config.json")):
        return model
    if model == DEFAULT_MODEL_DIR:
        model = DEFAULT_MODEL_ID
    local_dir = os.path.join(BASE_DIR, "models", model.split("/")[-1])
    if not os.path.isfile(os.path.join(local_dir, "config.json")):
        print(f"모델 다운로드: {model} -> {local_dir}")
        snapshot_download(repo_id=model, local_dir=local_dir)
    return local_dir
