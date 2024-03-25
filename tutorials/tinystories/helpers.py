# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

from docbuilder import TinyStoriesExtractor, TinyStoriesIterator


def write_jsonl(input_filename: str, output_dir: str, dump_every_n: int = 10000):
    """
    Convert a file to JSONL format and write it to the specified output directory.

    Args:
        input_filename (str): The path to the input file.
        output_dir (str): The path to the output directory.
        dump_every_n (int, optional): The number of records to dump to file at once.
    """
    if os.path.isdir(output_dir):
        if len(os.listdir(output_dir)) > 0:
            print(
                f"Output directory '{output_dir}' is not empty. Skipping conversion to JSONL."
            )
            return
    else:
        os.makedirs(output_dir)

    print(f"Converting '{input_filename}' to JSONL format (output: '{output_dir}')...")

    basename = os.path.basename(input_filename)
    iterator = TinyStoriesIterator()
    extractor = TinyStoriesExtractor()
    to_dump = []
    dump_ctr = 0

    def dump_to_file(to_dump, dump_ctr):
        """Helper function to facilitate dumping to file."""
        output_filename = f"{basename}-{dump_ctr}.jsonl"
        with open(os.path.join(output_dir, output_filename), "w") as output_file:
            output_file.writelines(to_dump)
        # Empty out the list and increment the counter.
        return [], dump_ctr + 1

    for item in iterator.iterate(input_filename):
        record_meta, content = item
        extracted = extractor.extract(content)

        if extracted is None:
            continue

        text_meta, text = extracted

        if text is None:
            continue

        line = {
            "text": text,
            **text_meta,
            **record_meta,
        }
        json_out = json.dumps(line, ensure_ascii=False)
        to_dump.append(json_out + "\n")

        # Should we dump what we have so far?
        if len(to_dump) == dump_every_n:
            to_dump, dump_ctr = dump_to_file(to_dump, dump_ctr)

    # Dump the remaining records.
    if to_dump:
        dump_to_file(to_dump, dump_ctr)
