"""Document processing and build orchestration for JSON output extension."""

import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from typing import List
from sphinx.application import Sphinx
from sphinx.util import logging

from ..core.builder import JSONOutputBuilder
from ..utils import get_setting, validate_content_gating_integration

logger = logging.getLogger(__name__)

def on_build_finished(app: Sphinx, exception: Exception) -> None:
    """Generate JSON files after HTML build is complete."""
    if exception is not None:
        return
    
    verbose = get_setting(app.config, 'verbose', False)
    log_func = logger.info if verbose else logger.debug
    
    log_func("Generating JSON output files...")
    
    # Validate content gating integration
    validate_content_gating_integration(app)
    
    try:
        json_builder = JSONOutputBuilder(app)
    except Exception as e:
        logger.error(f"Failed to initialize JSONOutputBuilder: {e}")
        return
    
    # Get all documents to process
    total_docs = len(app.env.all_docs)
    all_docs = []
    gated_docs = []
    
    for docname in app.env.all_docs:
        if json_builder.should_generate_json(docname):
            all_docs.append(docname)
        else:
            gated_docs.append(docname)
    
    if gated_docs:
        log_func(f"Content gating: excluding {len(gated_docs)} documents from JSON generation")
        if verbose and gated_docs:
            logger.debug(f"Gated documents: {', '.join(sorted(gated_docs))}")
    
    # Apply incremental build filtering
    if get_setting(app.config, 'incremental_build', False):
        incremental_docs = [docname for docname in all_docs if json_builder.needs_update(docname)]
        skipped_count = len(all_docs) - len(incremental_docs)
        if skipped_count > 0:
            log_func(f"Incremental build: skipping {skipped_count} unchanged files")
        all_docs = incremental_docs
    
    # Apply file size filtering if enabled
    skip_large_files = get_setting(app.config, 'skip_large_files', 0)
    if skip_large_files > 0:
        filtered_docs = []
        for docname in all_docs:
            try:
                source_path = app.env.doc2path(docname)
                if source_path and source_path.stat().st_size <= skip_large_files:
                    filtered_docs.append(docname)
                else:
                    log_func(f"Skipping large file: {docname} ({source_path.stat().st_size} bytes)")
            except Exception:
                filtered_docs.append(docname)  # Include if we can't check size
        all_docs = filtered_docs
    
    generated_count = 0
    failed_count = 0
    
    # Process documents in parallel if enabled
    if get_setting(app.config, 'parallel', False):
        generated_count, failed_count = process_documents_parallel(
            json_builder, all_docs, app.config, log_func
        )
    else:
        generated_count, failed_count = process_documents_sequential(
            json_builder, all_docs
        )
    
    log_func(f"Generated {generated_count} JSON files")
    if failed_count > 0:
        logger.warning(f"Failed to generate {failed_count} JSON files")

def process_documents_parallel(json_builder: JSONOutputBuilder, all_docs: List[str], 
                             config, log_func) -> tuple[int, int]:
    """Process documents in parallel batches."""
    parallel_workers = get_setting(config, 'parallel_workers', 'auto')
    if parallel_workers == 'auto':
        cpu_count = multiprocessing.cpu_count() or 1
        max_workers = min(cpu_count, 8)  # Limit to 8 threads max
    else:
        max_workers = min(int(parallel_workers), 16)  # Cap at 16 for safety
    
    batch_size = get_setting(config, 'batch_size', 50)
    
    generated_count = 0
    failed_count = 0
    
    # Process in batches to control memory usage
    for i in range(0, len(all_docs), batch_size):
        batch_docs = all_docs[i:i + batch_size]
        log_func(f"Processing batch {i//batch_size + 1}/{(len(all_docs)-1)//batch_size + 1} ({len(batch_docs)} docs)")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for docname in batch_docs:
                future = executor.submit(process_document, json_builder, docname)
                futures[future] = docname
            
            for future, docname in futures.items():
                try:
                    if future.result():
                        generated_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.error(f"Error generating JSON for {docname}: {e}")
                    failed_count += 1
    
    return generated_count, failed_count

def process_documents_sequential(json_builder: JSONOutputBuilder, all_docs: List[str]) -> tuple[int, int]:
    """Process documents sequentially."""
    generated_count = 0
    failed_count = 0
    
    for docname in all_docs:
        try:
            json_data = json_builder.build_json_data(docname)
            json_builder.write_json_file(docname, json_data)
            generated_count += 1
        except Exception as e:
            logger.error(f"Error generating JSON for {docname}: {e}")
            failed_count += 1
    
    return generated_count, failed_count

def process_document(json_builder: JSONOutputBuilder, docname: str) -> bool:
    """Process a single document for parallel execution."""
    try:
        json_data = json_builder.build_json_data(docname)
        json_builder.write_json_file(docname, json_data)
        json_builder.mark_updated(docname)  # Mark as processed for incremental builds
        return True
    except Exception as e:
        logger.error(f"Error generating JSON for {docname}: {e}")
        return False 