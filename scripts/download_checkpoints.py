"""Download the released checkpoints (README section 2) and record their HF revisions.

Fetches gridfm/genco-pf-datakit without its `mlflow/` run folders (thousands of small files,
not needed) and only the `pretraining/` folder of gridfm/genco-pf-transfer. Weights are never
committed to this repository; they are re-downloaded from these exact revisions.
"""
import json
import sys

from huggingface_hub import HfApi, snapshot_download

from common import CKPT_DIR, PRETRAIN_DIR, RESULTS_DIR, ROOT, load_settings


def main() -> int:
    hf = load_settings()["hf"]
    api = HfApi()
    revs = {}

    repo = hf["checkpoints_repo"]
    revs[repo] = api.model_info(repo).sha
    print(f"{repo}@{revs[repo][:8]} -> {CKPT_DIR.name}/")
    snapshot_download(repo, revision=revs[repo], local_dir=str(CKPT_DIR),
                      ignore_patterns=["mlflow/*"])

    repo = hf["pretraining_repo"]
    sub = hf["pretraining_subdir"]
    revs[repo] = api.model_info(repo).sha
    print(f"{repo}@{revs[repo][:8]} -> {PRETRAIN_DIR.parent.name}/{sub}/")
    snapshot_download(repo, revision=revs[repo], local_dir=str(PRETRAIN_DIR.parent),
                      allow_patterns=[f"{sub}/*"])

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "checkpoint_revisions.json").write_text(json.dumps(revs, indent=2))
    print(f"recorded revisions in {(RESULTS_DIR / 'checkpoint_revisions.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
