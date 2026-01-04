import pandas as pd
import os
import wikipediaapi
import re
import typer


BASE_PATH = "data/wiki_qa"
OUTPUT_ARTICLES_DIR = f"{BASE_PATH}/articles"
OUTPUT_KG_DIR = f"{BASE_PATH}/kgs"
DEFAULT_WIKIPEDIA_USER_AGENT = "MyWikiQADataFetcher/1.0 (example@example.com)"

wiki_api = wikipediaapi.Wikipedia(
    language="en", user_agent=DEFAULT_WIKIPEDIA_USER_AGENT
)


def download_and_save_split(split_name: str, base_output_path: str):
    os.makedirs(base_output_path, exist_ok=True)
    # TODO: load from hf instead? why only 00001 parquet?
    parquet_file_name = f"{split_name}-00000-of-00001.parquet"
    hf_path = f"hf://datasets/microsoft/wiki_qa/data/{parquet_file_name}"
    csv_file_path = os.path.join(base_output_path, f"{split_name}.csv")

    print(f"Downloading {split_name} data from {hf_path}...")
    try:
        df = pd.read_parquet(hf_path)
        print(f"Saving {split_name} data to {csv_file_path}...")
        df.to_csv(csv_file_path, index=False)
        print(
            f"{split_name.capitalize()} dataset successfully downloaded and saved to {csv_file_path}"
        )
        print(f"First 5 rows of {split_name}.csv:")
        print(df.head())
    except Exception as e:
        print(f"Error downloading or saving {split_name} data: {e}")


def download_all_wiki_qa_splits():
    splits_to_download = ["train", "test", "validation"]
    for split_name in splits_to_download:
        download_and_save_split(split_name, BASE_PATH)

    for split_name in splits_to_download:
        csv_file_path = os.path.join(BASE_PATH, f"{split_name}.csv")
        clean_file_path = os.path.join(BASE_PATH, f"{split_name}_clean.csv")

        try:
            df = pd.read_csv(csv_file_path)
            clean_df = df[df["label"] == 1]
            clean_df = clean_df.drop_duplicates(subset=["question_id"])
            clean_df.to_csv(clean_file_path, index=False)
            print(f"Clean {split_name} dataset saved to {clean_file_path}")
            print(f"Original: {len(df)} rows, Clean: {len(clean_df)} rows")
        except Exception as e:
            print(f"Error creating clean dataset for {split_name}: {e}")


def sanitize_filename(title: str) -> str:
    sanitized_title = re.sub(r"[^\w\s-]", "", title).strip()
    sanitized_title = re.sub(r"[-\s]+", "_", sanitized_title)
    sanitized_title = re.sub(r"_+", "_", sanitized_title)
    if not sanitized_title:
        sanitized_title = f"untitled_{abs(hash(title)) % (10**8)}"
    return sanitized_title


def retrieve_page(title: str) -> bool:
    try:
        page = wiki_api.page(title)
        if not page.exists():
            print(f'  Page "{title}" does not exist on Wikipedia. Skipping.')
            return False

        article_text = page.text
        pageid = page.pageid

        if not article_text:
            return False

        file_path = os.path.join(OUTPUT_ARTICLES_DIR, f"{sanitize_filename(title)}.txt")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(article_text)
        print(f'Saved version of "{title}" (pageid {pageid}) to {file_path}')
        return True

    except Exception as e:
        print(f'An error occurred while processing "{title}": {e}')
        return False


def main(split: str):
    if split not in ["train", "test", "validation"]:
        raise ValueError(f"Invalid split name: {split}")

    download_all_wiki_qa_splits()

    csv_path = os.path.join(BASE_PATH, f"{split}_clean.csv")
    df = pd.read_csv(csv_path)  # if "document_title" not in df.columns:
    raw_titles = df["document_title"].dropna().astype(str)
    document_titles = {title.strip() for title in raw_titles if title.strip()}
    print(f"Total unique, non-empty titles found in {csv_path}: {len(document_titles)}")

    if not document_titles:
        print("No document titles to process from CSV. Exiting article fetching.")
        return

    os.makedirs(OUTPUT_ARTICLES_DIR, exist_ok=True)
    fetched_count = 0

    for title in sorted(list(document_titles)):
        fetched_count += retrieve_page(title)

    print(f"Successfully fetched and saved {fetched_count} articles.")
    print(f"Skipped {len(document_titles) - fetched_count} articles.")


def _main():
    typer.run(main)


if __name__ == "__main__":
    _main()
