"""
Download the released trained motion model from GitHub Releases.

Weights are published as a GitHub Release asset (not stored in git) so the repo
stays lean and each model version is a separate, downloadable release. By default
this fetches the asset from the *latest* release, so you always get the newest
version. Needs no extra packages and no GitHub auth (public release).

Run:
    python scripts/download_model.py                 # latest version
    python scripts/download_model.py --tag motion-v1.0   # a specific version
"""
import argparse
import os
import sys
import urllib.request

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPTS_DIR, "..", "checkpoints")

REPO = "KesharaGunathilaka/adaptive-multimodal-hri"
# best_model_finetuned.pt (NTU-pretrained, fine-tuned on this project's real
# intent-video clips) is the deployed checkpoint (76.8% acc / 73.2% macro-F1
# on held-out test subjects).
ASSET = "best_model_finetuned.pt"
LATEST_URL = f"https://github.com/{REPO}/releases/latest/download/{ASSET}"


def _progress(block_num, block_size, total_size):
    done = block_num * block_size
    if total_size > 0:
        pct = min(100, done * 100 / total_size)
        sys.stdout.write(f"\r  {pct:5.1f}%  ({done/1e6:.1f}/{total_size/1e6:.1f} MB)")
    else:
        sys.stdout.write(f"\r  {done/1e6:.1f} MB")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="Release tag, e.g. motion-v1.0 (default: latest).")
    ap.add_argument("--url", default=None, help="Explicit asset URL (overrides --tag).")
    ap.add_argument("--out", default=os.path.join(CHECKPOINT_DIR, ASSET), help="Destination path.")
    ap.add_argument("--force", action="store_true", help="Re-download even if it exists.")
    args = ap.parse_args()

    if args.url:
        url = args.url
    elif args.tag:
        url = f"https://github.com/{REPO}/releases/download/{args.tag}/{ASSET}"
    else:
        url = LATEST_URL

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if os.path.exists(args.out) and not args.force:
        print(f"Already present: {args.out}\n(use --force to re-download)")
        return

    print(f"Downloading {url}")
    try:
        urllib.request.urlretrieve(url, args.out, _progress)
    except urllib.error.HTTPError as e:
        print(f"\nDownload failed ({e.code}). Make sure a release exists with asset "
              f"'{ASSET}'. Browse: https://github.com/{REPO}/releases")
        sys.exit(1)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
