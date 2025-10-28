import os
import pickle
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple

import click

PDFLATEX_BINARY_PATH = "/usr/bin/pdflatex"

TEX_PREAMBLE_ID = b"% NVpdftex LaTeX Preamble"
TEX_PREAMBLE_CODE = rb"""% NVpdftex LaTeX Preamble
% PHIL: The following code is used to always use T1 encoding and support .EPS images
\makeatletter%
\newcommand{\nvtexpybegin}{%
  \usepackage[T1]{fontenc}%
  \RequirePackage{epsfig}%
  \RequirePackage{epstopdf}%
  \g@addto@macro\Gin@extensions{,.EPS,.ps}%
  \epstopdfDeclareGraphicsRule{.EPS}{pdf}{.pdf}{%
    \ETE@epstopdf{##1}%
  }%
  \epstopdfDeclareGraphicsRule{.ps}{pdf}{.pdf}{%
    \ETE@epstopdf{##1}%
  }%
  \nvtexpyhook%
}%
\makeatother%
\AtBeginDocument{\nvtexpybegin}%
% PHIL: End of custom preamble, original tex code follows
"""

num_pdftex_calls = 4


@dataclass
class Page:
    """Represents a page in a document."""
    page_number: int
    png_path: Path
    pkl_path: Path


class ProcessingError(Exception):
    """Raised when processing fails."""
    pass


class FlatLayoutConversionError(Exception):
    """Raised when webdataset conversion fails."""
    pass


def removesuffix(s: str, suffix: str) -> str:
    """Remove suffix from string if present."""
    if s.endswith(suffix):
        return s[: -len(suffix)]
    return s


def pdftex_cmd(num_pdftex_call: int, tex_file_path: Path) -> list[str]:
    """Generate the pdflatex command for a given call number."""
    return [
        PDFLATEX_BINARY_PATH,
        *(
            [
                "-nvtexpy=8",
            ]
            if num_pdftex_call < num_pdftex_calls
            else []
        ),
        "-output-format",
        "pdf",
        "-interaction=batchmode",
        "-halt-on-error",
        str(tex_file_path),
    ]


def run_command(
    command: list[str],
    logpath: Path,
    workdir: Path,
    add_env: Optional[dict[str, str]] = None,
) -> None:
    """Run a command and log its output."""
    with logpath.open("w", errors="backslashreplace") as logfile:
        start = time.perf_counter()
        try:
            subprocess.run(
                " ".join(command),
                input="",
                stdout=logfile,
                stderr=logfile,
                cwd=workdir,
                check=True,
                shell=True,
                encoding="utf-8",
                errors="backslashreplace",
                env=os.environ | add_env if add_env is not None else None,
            )
        finally:
            elapsed = time.perf_counter() - start
            if elapsed > 60:
                print(f"Command took {elapsed:.2f}s: {' '.join(command)}")


def find_pages(path: Path) -> Generator[Tuple[int, Path], None, None]:
    """Find all generated page PNG files."""
    page_num = 1
    while True:
        png_path = path / f"nvpdftex_page_{page_num}.png"
        if png_path.exists():
            yield page_num, png_path
            page_num += 1
        else:
            break


def convert_to_flat_layout(pkl_path: Path) -> str:
    """
    Convert a pickle file to flat layout format.
    
    Args:
        pkl_path: Path to the .pkl file
        
    Returns:
        JSON string in flat layout format
    """
    from nvtexpy.output.flat_layout import convert_from_pkl as convert_flat_layout  # type: ignore
    
    with open(pkl_path, "rb") as rf:
        pkl_data = pickle.load(rf)
    
    try:
        flat_layout_output = convert_flat_layout(
            pkl_data, 
            table_format="latex", 
        )
        return flat_layout_output
    except Exception as e:
        # print(f"*** error converting {pkl_path}: {type(e).__name__}: {e}")
        # traceback.print_exc(file=sys.stdout)
        raise FlatLayoutConversionError(f"Flat layout conversion failed: {e}") from e


def process_tex_file(
    tex_file_path: Path,
) -> list[Page]:
    """
    Process a LaTeX file to generate PNG images of its pages.
    
    Args:
        tex_file_path: Path to the .tex file to process
        
    Returns:
        List of SimplePage objects representing the generated pages
    """
    start_time = time.perf_counter()
    
    # Work in the same directory as the tex file
    work_dir = tex_file_path.parent
    basename = removesuffix(tex_file_path.name, ".tex")
    pdf_file_path = work_dir / f"{basename}.pdf"
    
    print(f"Processing {tex_file_path}")
    
    # Step 1: Add preamble
    print("Adding preamble...")
    file_bytes = tex_file_path.read_bytes()
    if file_bytes.startswith(TEX_PREAMBLE_ID):
        print("Skipping - preamble already present")
    else:
        with tex_file_path.open("wb") as texwf:
            texwf.write(TEX_PREAMBLE_CODE)
            texwf.write(file_bytes)
    
    # Step 2: Run pdflatex multiple times
    for i_pdftex_call in range(num_pdftex_calls):
        print(f"Running pdflatex pass {i_pdftex_call + 1}/{num_pdftex_calls}...")
        step_log_path = work_dir / f"pdftex_{i_pdftex_call + 1}.log"
        
        run_command(
            pdftex_cmd(i_pdftex_call + 1, tex_file_path),
            logpath=step_log_path,
            workdir=work_dir,
            add_env={
                # Move tex font cache etc folders away from home directory
                "TEXMFHOME": "/tmp/texmf",
                "TEXMFCONFIG": "/tmp/texmf-config",
                "TEXMFVAR": "/tmp/texmf-var",
            },
        )
    
    # Step 3: Run Ghostscript to convert PDF to PNG
    print("Converting PDF to PNG images...")
    run_command(
        [
            "gs",
            "-dNOPAUSE",
            "-dBATCH",
            "-sDEVICE=png16m",
            "-dTextAlphaBits=4",
            "-dGraphicsAlphaBits=4",
            "-dMaxBitmap=500000000",
            "-dAlignToPixels=0",
            "-dGridFitTT=2",
            "-r300",  # 300 DPI
            "-o",
            str(work_dir / "nvpdftex_page_%d.png"),
            str(pdf_file_path),
        ],
        logpath=work_dir / "gs.log",
        workdir=work_dir,
    )
    
    # Step 4: Find and create page objects
    print("Finding generated pages...")
    pages: list[Page] = []
    for page_num, png_path in find_pages(work_dir):
        pkl_path = work_dir / f"nvpdftex_page_{page_num}.pkl"
        if not pkl_path.exists():
            print(f"WARNING: Missing pkl file for page {page_num}")
            continue
            
        page = Page(
            page_number=page_num,
            png_path=png_path,
            pkl_path=pkl_path,
        )
        pages.append(page)
        print(f"  Page {page_num}: {png_path}")
    
    if not pages:
        raise ProcessingError("No pages were generated!")
    
    # Step 5: Optionally convert to flat layout format
    print("Converting pages to flat layout format...")
    for page in pages:
        try:
            flat_layout_json = convert_to_flat_layout(page.pkl_path)
            flat_layout_path = page.pkl_path.with_suffix(".flat_layout.json")
            with open(flat_layout_path, "w") as f:
                f.write(flat_layout_json)
            print(f"  Page {page.page_number}: {flat_layout_path}")
        except FlatLayoutConversionError as e:
            print(f"  Page {page.page_number}: Conversion failed - {e}")

    elapsed = time.perf_counter() - start_time
    print(f"Completed in {elapsed:.1f} seconds")
    
    return pages


@click.command()
@click.argument("tex_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def main(tex_file: Path) -> None:
    """
    Process a single LaTeX file to generate PNG images.
    
    TEX_FILE: Path to the .tex file to process
    """
    try:
        pages = process_tex_file(
            tex_file.absolute(),
        )
        print(f"\nSuccessfully processed {len(pages)} page(s)")
    except Exception as e:
        print(f"Error processing {tex_file}: {e}")
        raise


if __name__ == "__main__":
    main()

