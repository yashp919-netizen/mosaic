"""Push Mosaic to a HuggingFace Space.

Usage:
    1. huggingface-cli login        # one-time — paste your HF token
    2. python deploy/push_to_hf.py  # creates/updates the Space

The Space is created as: https://huggingface.co/spaces/<your-HF-username>/mosaic
"""

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent

HF_SPACE_ID = "yashp919-netizen/mosaic"   # change if your HF username differs


def main() -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("huggingface_hub not installed — run: pip install huggingface_hub")

    api = HfApi()

    try:
        user = api.whoami()["name"]
        print(f"Logged in as: {user}")
    except Exception:
        sys.exit(
            "Not logged in to HuggingFace.\n"
            "Run: huggingface-cli login\n"
            "Then re-run this script."
        )

    # Create the Space if it doesn't exist
    try:
        api.create_repo(
            repo_id=HF_SPACE_ID,
            repo_type="space",
            space_sdk="streamlit",
            exist_ok=True,
            private=False,
        )
        print(f"Space ready: https://huggingface.co/spaces/{HF_SPACE_ID}")
    except Exception as e:
        sys.exit(f"Could not create Space: {e}")

    # Build a flat staging directory for the Space
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Space config README (with YAML frontmatter)
        shutil.copy(ROOT / "deploy" / "hf_space" / "README.md", tmp_path / "README.md")

        # requirements.txt
        shutil.copy(ROOT / "deploy" / "hf_space" / "requirements.txt", tmp_path / "requirements.txt")

        # The Streamlit app (renamed to app.py for HF Spaces convention)
        shutil.copy(ROOT / "ui" / "streamlit_app.py", tmp_path / "app.py")

        # Source package
        shutil.copytree(ROOT / "src", tmp_path / "src")

        # Data (generator + schema; not the CSVs — generated at runtime)
        (tmp_path / "data" / "synth").mkdir(parents=True)
        shutil.copy(ROOT / "data" / "synth" / "generator.py", tmp_path / "data" / "synth" / "generator.py")
        shutil.copy(ROOT / "data" / "target_schema.yaml", tmp_path / "data" / "target_schema.yaml")

        # Startup script — generate data if CSVs are missing
        startup = tmp_path / "startup.sh"
        startup.write_text(
            "#!/bin/bash\n"
            "if [ ! -f data/synth/market_uk.csv ]; then\n"
            "  python data/synth/generator.py --output-dir data/synth/\n"
            "fi\n"
            "streamlit run app.py --server.address 0.0.0.0 --server.port 7860\n"
        )

        # Upload the whole staging directory
        print("Uploading to HuggingFace Space ...")
        api.upload_folder(
            repo_id=HF_SPACE_ID,
            repo_type="space",
            folder_path=str(tmp_path),
            commit_message="Deploy Mosaic v1 to HuggingFace Spaces",
        )

    space_url = f"https://huggingface.co/spaces/{HF_SPACE_ID}"
    print(f"\nDone! Space live at: {space_url}")
    print("\nNext steps:")
    print("  1. Open the Space URL above and verify it loads")
    print("  2. Run the UK market end-to-end to confirm it works")
    print("  3. (Optional) Add GEMINI_API_KEY in Space Settings → Secrets")
    print("  4. Add the Space URL to README under the Demo section")

    return space_url


if __name__ == "__main__":
    main()
