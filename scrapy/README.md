# Scrapy Web Scraping Project

This directory contains Scrapy spiders for collecting academic data from various university websites.

## Project Structure

```
scrapy/
├── README.md                 # This file
├── scrapy.cfg               # Scrapy configuration
├── scrapy_data/             # Main Scrapy project
│   ├── __init__.py
│   ├── items.py             # Data models
│   ├── middlewares.py       # Custom middlewares
│   ├── pipelines.py         # Data processing pipelines
│   ├── settings.py          # Scrapy settings
│   └── spiders/             # Spider definitions
│       ├── __init__.py
│       ├── cs.py            # Computer Science department scraper
│       ├── engishclasses.py # English department scraper
│       ├── ugabulletin.py   # UGA Bulletin scraper
│       └── uga_rmp.py       # RateMyProfessors UGA scraper
└── output/                  # Scraped data output
    ├── raw_data/            # Raw scraped data (JSON files)
```

## Spiders

### 1. UGA RateMyProfessors (`uga_rmp.py`)
- **Purpose**: Scrapes all University of Georgia professors from RateMyProfessors
- **Data**: Professor names, ratings, difficulty scores, and all reviews with dates and tags
- **Output**: `output/raw_data/uga_rmp.json`
- **Status**: ✅ **Complete** - Scraped 5,144 professors with reviews

### 2. Computer Science Department (`cs.py`)
- **Purpose**: Scrapes UGA Computer Science department pages
- **Data**: Course information, faculty details, program requirements
- **Output**: `output/raw_data/cs_uga.json`

### 3. English Department (`engishclasses.py`)
- **Purpose**: Scrapes UGA English department pages
- **Data**: Course information, faculty details, program requirements
- **Output**: `output/raw_data/english_uga.json`

### 4. UGA Bulletin (`ugabulletin.py`)
- **Purpose**: Scrapes UGA Bulletin course catalog
- **Data**: Course descriptions, prerequisites, credit hours
- **Output**: `output/raw_data/ugabulletin.json`

## Usage

### Running Spiders

From the `scrapy` directory:

```bash
# Run all spiders
scrapy list | xargs -I {} scrapy crawl {}

# Run specific spider
scrapy crawl uga_rmp
scrapy crawl cs
scrapy crawl engishclasses
scrapy crawl ugabulletin
```

### Data Processing

After scraping, run the data cleaning pipeline:

```bash
cd ..  # Go back to project root
python pipeline/clean_data.py
```

This will process the raw scraped data and create cleaned versions in `cleaned_data_csv/`.

## Configuration

- **Download Delay**: 1 second between requests (polite scraping)
- **User Agent**: Realistic browser user agent
- **Output Format**: JSON with UTF-8 encoding
- **Error Handling**: Graceful handling of timeouts and missing data

## Data Quality

- **RateMyProfessors**: 5,144 professors scraped successfully
- **Error Rate**: Very low (< 0.1% failed requests)
- **Completeness**: Full professor profiles with all available reviews
- **Data Integrity**: Proper JSON structure with consistent field names

## Notes

- The RateMyProfessors spider uses GraphQL API for efficient data collection
- All spiders include proper error handling and logging
- Data is automatically organized by source and date
- Raw data is preserved for reprocessing if needed

## Next Steps

1. Run data cleaning pipeline on new RateMyProfessors data
2. Integrate with existing pipeline for QA generation
3. Consider adding more university departments
4. Implement data validation and quality checks 