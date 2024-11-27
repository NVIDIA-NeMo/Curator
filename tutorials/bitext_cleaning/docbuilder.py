import os

import requests

from nemo_curator.download.doc_builder import DocumentDownloader


class TedTalksDownloader(DocumentDownloader):
    def __init__(self, download_dir: str):
        super().__init__()

        if not os.path.isdir(download_dir):
            os.makedirs(download_dir)

        self._download_dir = download_dir
        print("Download directory: ", self._download_dir)

    def download(self, url_src: str, url_tgt: str, force=False) -> str:
        src_filename = os.path.basename(url_src)
        src_out = os.path.join(self._download_dir, src_filename)
        tgt_filename = os.path.basename(url_tgt)
        tgt_out = os.path.join(self._download_dir, tgt_filename)

        sides = ["src", "tgt"]
        urls = [url_src, url_tgt]
        output_files = {"src": src_out, "tgt": tgt_out}
        for side, url, out_file_key in zip(sides, urls, output_files):
            if os.path.exists(output_files[out_file_key]) and force is False:
                print(
                    f"File '{output_files[out_file_key]}' already exists, skipping download."
                )
            else:
                print(f"Downloading TED Talks dataset from '{url}'...")
                response = requests.get(url)

                with open(output_files[out_file_key], "wb") as file:
                    file.write(response.content)

        return output_files
