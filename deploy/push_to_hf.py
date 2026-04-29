"""Push Mosaic to a HuggingFace Space (Docker SDK).

Usage:
    python deploy/push_to_hf.py --token hf_xxxx
    # or set HF_TOKEN env var and run without --token
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
HF_SPACE_ID = "yashp919/mosaic"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    args = parser.parse_args()

    token = args.token
    if not token:
        sys.exit("Provide --token hf_xxxx or set HF_TOKEN env var.")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("huggingface_hub not installed — run: pip install huggingface_hub")

    api = HfApi(token=token)

    try:
        user = api.whoami()["name"]
        print(f"Logged in as: {user}")
    except Exception as e:
        sys.exit(f"Auth failed: {e}")

    # Create the Space (docker SDK — streamlit is no longer a direct SDK option)
    try:
        api.create_repo(
            repo_id=HF_SPACE_ID,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
            private=False,
        )
        print(f"Space ready: https://huggingface.co/spaces/{HF_SPACE_ID}")
    except Exception as e:
        sys.exit(f"Could not create Space: {e}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Space config README (YAML frontmatter tells HF it's a Docker Space)
        shutil.copy(ROOT / "deploy" / "hf_space" / "README.md",    tmp_path / "README.md")
        shutil.copy(ROOT / "deploy" / "hf_space" / "Dockerfile",    tmp_path / "Dockerfile")
        shutil.copy(ROOT / "deploy" / "hf_space" / "requirements.txt", tmp_path / "requirements.txt")

        # Streamlit app (app.py is the entrypoint referenced in Dockerfile CMD)
        shutil.copy(ROOT / "ui" / "streamlit_app.py", tmp_path / "app.py")

        # Source package
        shutil.copytree(ROOT / "src", tmp_path / "src")

        # Data (generator + schema; CSVs generated at image build time)
        (tmp_path / "data" / "synth").mkdir(parents=True)
        shutil.copy(ROOT / "data" / "synth" / "generator.py",
                    tmp_path / "data" / "synth" / "generator.py")
        shutil.copy(ROOT / "data" / "target_schema.yaml",
                    tmp_path / "data" / "target_schema.yaml")

        print("Uploading to HuggingFace Space ...")
        api.upload_folder(
            repo_id=HF_SPACE_ID,
            repo_type="space",
            folder_path=str(tmp_path),
            commit_message="Deploy Mosaic v1 — Docker SDK",
        )

    space_url = f"https://huggingface.co/spaces/{HF_SPACE_ID}"
    print(f"\nDone! Space live at: {space_url}")
    print("Build takes ~3–5 min. Refresh the Space URL to check progress.")
    print("Optional: add GEMINI_API_KEY in Space Settings > Secrets to enable Gemini path.")


if __name__ == "__main__":
    main()
