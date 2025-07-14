import json
from dataclasses import dataclass
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from loguru import logger

from ray_curator.stages.download.text import URLGenerator

# Request timeout in seconds
REQUEST_TIMEOUT = 30


@dataclass
class WikipediaUrlGenerator(URLGenerator):
    """Generates URLs for Wikipedia dump files."""

    language: str = "en"
    dump_date: str | None = None
    wikidumps_index_prefix: str = "https://dumps.wikimedia.org"

    def generate_urls(self) -> list[str]:
        """Generate Wikipedia dump URLs.

        Returns:
            List of URLs pointing to Wikipedia dump files
        """
        return self._get_wikipedia_urls()

    def _get_wikipedia_urls(self) -> list[str]:
        """
        Retrieves all URLs pointing to Wikipedia dumps for the specified language and date.

        Returns:
            List of URLs for Wikipedia dump files
        """
        wiki_index_url = urljoin(self.wikidumps_index_prefix, f"{self.language}wiki")

        dump_date = self.dump_date
        if not dump_date:
            # Get the latest dump date from the index
            logger.info(f"Fetching latest dump date from {wiki_index_url}")
            raw_wiki_index = requests.get(wiki_index_url, timeout=REQUEST_TIMEOUT)
            wiki_index = raw_wiki_index.content.decode("utf-8")
            wiki_index_parsed = BeautifulSoup(wiki_index, "lxml")

            # Get all dumps available in the index
            dumps = wiki_index_parsed.find_all("a")
            # TODO: Can Either Sarah or Praateek help me with this? If no one is aware about Wikipedia, happy to take it later
            # Ideally this should be dumps[-2].
            # Dumps is something like:
            # (Pdb) p dumps
            # [<a href="../">../</a>, <a href="20250401/">20250401/</a>, <a href="20250420/">20250420/</a>,
            # <a href="20250501/">20250501/</a>, <a href="20250520/">20250520/</a>, <a href="20250601/">20250601/</a>,
            # <a href="20250620/">20250620/</a>, <a href="20250701/">20250701/</a>, <a href="latest/">latest/</a>]
            # But with the latest dump of -2, I get no files in dump_data["jobs"]["articlesmultistreamdump"]["files"].
            # With the satutus being waiting.
            # This problem is same in nemo_curator.utils.download_utils.get_wikipedia_urls(), so not specific to ray-curator.
            dump_date = dumps[-3].text
            logger.info(f"Found latest dump date: {dump_date}")
        else:
            # A trailing / is needed for the URL
            dump_date = dump_date + "/"

        # Get the JSON dump data
        wiki_latest_dump = urljoin(wiki_index_url + "/", dump_date)
        wiki_latest_dump_status = urljoin(wiki_latest_dump, "dumpstatus.json")

        logger.info(f"Fetching dump status from {wiki_latest_dump_status}")
        raw_dump_data = requests.get(wiki_latest_dump_status, timeout=REQUEST_TIMEOUT)

        try:
            dump_data = json.loads(raw_dump_data.content)
        except json.JSONDecodeError:
            clean_dump_date = dump_date.removesuffix("/")
            msg = f"No Wikipedia dump found for {clean_dump_date}"
            raise ValueError(msg) from None

        # Get all multistream files within the dump data
        wikipedia_urls = []
        for file_name in dump_data["jobs"]["articlesmultistreamdump"]["files"]:
            if "xml" in file_name:
                url = urljoin(wiki_latest_dump, file_name)
                wikipedia_urls.append(url)

        logger.info(f"Found {len(wikipedia_urls)} Wikipedia dump files")
        return wikipedia_urls
