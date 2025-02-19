import requests
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Dict, List, Tuple


class ImageDownloader:
    def __init__(self, api_url: str, output_dir: str):
        self.api_url = api_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def fetch_image_urls(self) -> Dict[str, List[str]]:
        """Fetch all image URLs from the API"""
        try:
            response = requests.post(
                self.api_url,
                json={"productIDs": []}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to fetch image URLs: {e}")
            return {}

    def download_image(self, url_info: Tuple[str, str, int]) -> bool:
        """Download a single image"""
        product_id, url, image_index = url_info
        try:
            if not url:
                return False

            product_dir = self.output_dir / product_id
            product_dir.mkdir(exist_ok=True)

            filename = f"{image_index}.jpg"
            filepath = product_dir / filename

            if filepath.exists():
                self.logger.info(f"Skipping existing file: {product_id}/{filename}")
                return True

            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            self.logger.info(f"Successfully downloaded: {product_id}/{filename}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to download image {url}: {e}")
            return False

    def download_dataset(self, max_products=None, num_threads=5):
        """Download all images using multiple threads"""
        self.logger.info("Fetching image URLs...")
        product_data = self.fetch_image_urls()

        if max_products:
            product_data = dict(list(product_data.items())[:max_products])

        download_tasks = []
        for product_id, urls in product_data.items():
            for idx, url in enumerate(urls):
                download_tasks.append((product_id, url, idx))

        total_images = len(download_tasks)
        self.logger.info(f"Found {total_images} images to download from {len(product_data)} products")

        successful = 0
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(self.download_image, download_tasks))
            successful = sum(results)

        self.logger.info(f"Download complete. Successfully downloaded {successful}/{total_images} images")

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(product_data, f, indent=2)

        return successful


def main():
    API_URL = "http://api.athebyme-market.ru:8081/api/media"
    OUTPUT_DIR = "dataset/raw_images"
    MAX_PRODUCTS = 5000

    downloader = ImageDownloader(API_URL, OUTPUT_DIR)
    downloader.download_dataset(max_products=MAX_PRODUCTS)


if __name__ == "__main__":
    main()