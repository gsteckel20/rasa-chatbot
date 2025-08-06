# Incremental Data Processing: Future Plan

## Problem
As the dataset grows and new scraped data or chat logs are added, it becomes inefficient to reprocess all data every time the pipeline is run. Ideally, we want to process only new or changed data, not everything.

## Simple Solution (for Now)
Currently, the pipeline processes all data in the input folders (e.g., `scrapy/output`, `groupme_data`). This is fine for small datasets or early development, but will become slow and wasteful as data grows.

## Proposed Future Solution: `data_to_process` Folder
- **Idea:** When new data is scraped or collected, move/copy it into a `data_to_process` folder.
- The pipeline only processes files in `data_to_process`.
- After processing, move the files to an archive or mark them as processed.
- This avoids reprocessing old data and keeps the workflow simple.

## Alternative: Manifest/Log Approach
- Maintain a manifest (JSON, CSV, or lightweight DB) that records which items/files have already been processed (using unique IDs, hashes, or timestamps).
- On each run, only process new or changed items.
- Update the manifest after processing.

## Pros/Cons
| Approach              | Pros                        | Cons                        |
|-----------------------|-----------------------------|-----------------------------|
| `data_to_process` dir | Simple, easy to manage      | Manual file management      |
| Manifest/log          | Fully automated, scalable   | Needs careful ID tracking   |

## Summary
For now, processing all data is acceptable. As the project grows, consider switching to a `data_to_process` folder or a manifest/log-based incremental processing system to save time and resources. 