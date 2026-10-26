"""Back up results/ to a private Hugging Face dataset repo, or restore it (works on Colab and laptop).

Run:  python -m src.hub push    # upload results/ (creates the private repo the first time)
      python -m src.hub pull    # download everything into results/

Token: read from the HF_TOKEN environment variable, otherwise asked for. Never put it in code or git.
"""
import argparse
import os
from getpass import getpass

from huggingface_hub import HfApi, snapshot_download

from src.config import HF_REPO_NAME, RESULTS


def connect():
    """Return (api, repo_id, token). The repo owner is the account the token belongs to."""
    token = os.environ.get("HF_TOKEN") or getpass("Hugging Face token (write access): ")
    api = HfApi(token=token)
    repo_id = f"{api.whoami()['name']}/{HF_REPO_NAME}"
    return api, repo_id, token


def push():
    api, repo_id, _ = connect()
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(folder_path=RESULTS, repo_id=repo_id, repo_type="dataset",
                      ignore_patterns=[".gitkeep", ".cache/*"], commit_message="backup results")
    print(f"pushed {RESULTS} -> https://huggingface.co/datasets/{repo_id}")


def pull():
    _, repo_id, token = connect()
    snapshot_download(repo_id, repo_type="dataset", local_dir=RESULTS, token=token)
    print(f"pulled {repo_id} -> {RESULTS}:", sorted(p.name for p in RESULTS.iterdir()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["push", "pull"])
    {"push": push, "pull": pull}[parser.parse_args().action]()
