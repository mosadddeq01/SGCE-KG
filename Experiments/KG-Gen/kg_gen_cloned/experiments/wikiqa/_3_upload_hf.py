import os
import sys
from pathlib import Path


try:
    from huggingface_hub import HfApi, create_repo
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install required packages:")
    print("  pip install huggingface_hub tqdm")
    sys.exit(1)


def upload_wikiqa_dataset(dry_run=False, skip_confirm=False, api_key=None):
    """Upload the WikiQA dataset to HuggingFace Hub."""

    # Configuration
    HF_TOKEN = api_key or os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        print("HF_TOKEN is not set, run huggingface-cli login")
        sys.exit(1)

    REPO_NAME = "wikiqa-kggen-dataset"
    REPO_TYPE = "dataset"
    ORGANIZATION = None  # ORG NAME?

    if dry_run:
        print("DRY RUN MODE - No actual upload will be performed")

    # Local data path
    data_path = Path("data/wiki_qa").resolve()
    if not data_path.exists():
        print(f"Error: Data path {data_path} does not exist")
        sys.exit(1)

    print(f"Uploading data from: {data_path}")
    print(f"Repository: {REPO_NAME}")
    print(f"Repository type: {REPO_TYPE}")
    api = HfApi(token=HF_TOKEN)

    # Create repository if it doesn't exist
    try:
        if ORGANIZATION:
            repo_id = f"{ORGANIZATION}/{REPO_NAME}"
        else:
            user_info = api.whoami()
            username = user_info["name"]
            repo_id = f"{username}/{REPO_NAME}"

        print(f"Creating/checking repository: {repo_id}")

        try:
            api.repo_info(repo_id, repo_type=REPO_TYPE)
            print(f"Repository {repo_id} already exists")
        except:  # noqa: E722
            # Create the repository
            create_repo(
                repo_id=repo_id,
                repo_type=REPO_TYPE,
                token=HF_TOKEN,
                private=False,
                exist_ok=True,
            )
            print(f"Created repository: {repo_id}")

    except Exception as e:
        print(f"Error setting up repository: {e}")
        sys.exit(1)

    # Collect all files to upload
    files_to_upload = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            full_path = Path(root) / file
            # Get relative path from data directory
            rel_path = full_path.relative_to(data_path.parent.parent)
            files_to_upload.append((full_path, rel_path))

    print(f"Found {len(files_to_upload)} files to upload")

    # Upload files with progress bar
    if dry_run:
        print(f"Would upload {len(files_to_upload)} files to {repo_id}")
        for local_path, repo_path in files_to_upload[:5]:  # Show first 5 files
            print(f"  {local_path} -> {repo_path}")
        if len(files_to_upload) > 5:
            print(f"  ... and {len(files_to_upload) - 5} more files")
        return

    # Confirm upload
    if not skip_confirm:
        response = input(
            f"\nReady to upload {len(files_to_upload)} files to https://huggingface.co/datasets/{repo_id}\nContinue? (y/N): "
        )
        if response.lower() not in ["y", "yes"]:
            print("Upload cancelled.")
            return

    print("Starting upload...")
    with tqdm(total=len(files_to_upload), desc="Uploading files") as pbar:
        for local_path, repo_path in files_to_upload:
            try:
                api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=str(repo_path),
                    repo_id=repo_id,
                    repo_type=REPO_TYPE,
                    token=HF_TOKEN,
                )
                pbar.set_postfix(file=str(repo_path))
            except Exception as e:
                print(f"Error uploading {local_path}: {e}")
            pbar.update(1)

    print("\nUpload completed!")
    print(f"Dataset available at: https://huggingface.co/datasets/{repo_id}")

    # Create/update README with dataset description
    readme_content = """---
dataset_info:
  features:
    - name: articles
      dtype: string
    - name: knowledge_graphs
      dtype: dict
    - name: qa_splits
      dtype: dict
  config_name: wikiqa
  version: 1.0.0
  splits:
    - name: train
      num_examples: 87360
    - name: validation
      num_examples: 10570
    - name: test
      num_examples: 2960
  download_size: 500000000
  dataset_size: 500000000
---

# WikiQA Knowledge Graphs Dataset

This dataset contains Wikipedia articles paired with automatically generated knowledge graphs for question answering research.

## Dataset Structure

- `articles/` - Original Wikipedia articles in text format
- `articles_*_kg/` - Knowledge graphs generated for different article subsets in JSON format
- `aggregated/` - Aggregated knowledge graphs and statistics
- `*.csv` - Question answering train/validation/test splits

## Subsets

- `articles_1/` - Single article (2006 Winter Olympics)
- `articles_40k_ch/` - ~40k character chunks
- `articles_400k_ch/` - ~400k character chunks
- `articles_4m_ch/` - ~4M character chunks
- `articles_20m_ch/` - ~20M character chunks
- `articles_w_context/` - Articles with additional context

## Knowledge Graph Generation

Knowledge graphs were generated using the KG-Gen library with entity extraction, relation extraction, and graph clustering steps.

## Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{{kg-gen-wikiqa,
  title={{WikiQA Knowledge Graphs Dataset}},
  author={{KG-Gen Research Team}},
  year={{2024}}
}}
```
"""

    # Upload README
    try:
        api.upload_file(
            path_or_fileobj=readme_content,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type=REPO_TYPE,
            token=HF_TOKEN,
        )
        print("README.md uploaded successfully")
    except Exception as e:
        print(f"Error uploading README: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload WikiQA dataset to HuggingFace Hub"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading",
    )
    parser.add_argument(
        "--api-key",
        action="store",
        help="Organization name to upload to",
    )
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    if args.yes and args.dry_run:
        print("Cannot use --yes with --dry-run")
        sys.exit(1)

    upload_wikiqa_dataset(
        dry_run=args.dry_run, skip_confirm=args.yes, api_key=args.api_key
    )
