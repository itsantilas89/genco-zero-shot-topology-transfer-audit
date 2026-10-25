"""Write results/provenance.json and results/SHA256SUMS (README section 12, items 3-4)."""
import hashlib
import json
import platform
import subprocess
import sys

from common import DATA_DIR, GRAPHKIT_DIR, RESULTS_DIR, ROOT


def sh(*cmd, cwd=ROOT) -> str:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                          # noqa: BLE001
        return ""


def main() -> int:
    prov = {
        "graphkit_commit": sh("git", "rev-parse", "HEAD", cwd=GRAPHKIT_DIR),
        "audit_repo_commit": sh("git", "rev-parse", "HEAD"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": sh("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"),
        "cuda": next((l.strip() for l in sh("nvidia-smi").splitlines() if "CUDA Version" in l), ""),
        "checkpoint_revisions": json.loads((RESULTS_DIR / "checkpoint_revisions.json").read_text())
        if (RESULTS_DIR / "checkpoint_revisions.json").exists() else {},
        "datasets": [json.loads(p.read_text()) for p in sorted(DATA_DIR.glob("*/n_scenarios_present.json"))],
    }
    try:
        import torch
        prov["torch"] = torch.__version__
        prov["torch_cuda"] = torch.version.cuda
    except ImportError:
        pass
    (RESULTS_DIR / "provenance.json").write_text(json.dumps(prov, indent=2))

    lines = []
    for p in sorted(RESULTS_DIR.glob("*.csv")) + sorted((RESULTS_DIR / "analyses").glob("*.csv")):
        lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}")
    (RESULTS_DIR / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    print(f"wrote provenance.json and SHA256SUMS ({len(lines)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
