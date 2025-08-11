# Instantly API Query Tool

This tool queries the Instantly API and exports all available contact data to a CSV file with comprehensive column analysis.

## 🚀 Features

- **Complete Data Export**: Retrieves all available columns from Instantly API
- **JSONB Flattening**: Automatically flattens nested JSONB structures into additional columns
- **Comprehensive Analysis**: Analyzes contact structure and provides detailed column summary
- **Flexible Querying**: Support for search queries and campaign-specific data
- **Error Handling**: Robust error handling with detailed logging

## 📋 Prerequisites

1. **Instantly API Key**: Get your API key from your Instantly dashboard
2. **Python Dependencies**: Install required packages

```bash
pip install requests pandas
```

## 🔧 Usage

### Method 1: Simple Script (Recommended)

1. **Edit the API key** in `run_instantly_query.py`:
```python
API_KEY = "your_actual_api_key_here"
```

2. **Run the script**:
```bash
python run_instantly_query.py
```

### Method 2: Command Line Interface

```bash
python instantly_api_query.py --api-key YOUR_API_KEY --output contacts.csv --limit 1000
```

**Parameters**:
- `--api-key`: Your Instantly API key (required)
- `--output`: Output CSV filename (default: instantly_contacts_export.csv)
- `--limit`: Number of contacts to retrieve (default: 1000)
- `--search`: Optional search query to filter contacts

### Method 3: Python Import

```python
from instantly_api_query import query_instantly_api

# Query all contacts
df = query_instantly_api(
    api_key="your_api_key",
    output_file="contacts.csv",
    limit=1000
)

# Search for specific contacts
df = query_instantly_api(
    api_key="your_api_key",
    output_file="ceo_contacts.csv",
    limit=500,
    search_query="CEO"
)
```

## 📊 Output

The script generates:

1. **CSV File**: Complete contact data with all available columns
2. **Console Summary**: Detailed analysis of retrieved data
3. **Column Analysis**: Breakdown of all columns and their data completeness

### Example Output Summary:
```
============================================================
INSTANTLY API QUERY SUMMARY
============================================================
📊 Total contacts: 1,000
🔧 Total columns: 45
📁 Output file: instantly_contacts_export.csv
📏 File size: 2.3 MB

📋 Column Summary:
    1. id                           | Non-null: 1000 | Null:    0
    2. email                        | Non-null:  950 | Null:   50
    3. first_name                   | Non-null:  800 | Null:  200
    4. last_name                    | Non-null:  850 | Null:  150
    5. title                        | Non-null:  900 | Null:  100
    ...

🔍 JSONB Structure Details:
  employment_history: id, title, company, start_date, end_date, current, ...
  organization_data: id, name, industry, website, phone, employees, ...
  account_data: id, domain, source, team_id, owner_id, ...
  api_response_raw: id, email, title, account, functions, departments, ...
```

## 🔍 API Endpoints Used

The script queries these Instantly API endpoints:

- **GET /api/v1/contacts**: Retrieve all contacts
- **GET /api/v1/contacts/search**: Search contacts with query
- **GET /api/v1/campaigns/{id}/contacts**: Get contacts from specific campaign
- **GET /api/v1/contacts/{id}**: Get detailed contact information

## 📈 Data Structure

### Standard Columns
- Basic contact info (id, email, name, title)
- Campaign data (campaign_id, status, engagement metrics)
- Timestamp data (created, updated, last_contact)
- Geographic data (country, city, state)

### JSONB Columns (Automatically Flattened)
- **employment_history**: Job history and career progression
- **organization_data**: Company information and details
- **account_data**: Account management and CRM data
- **api_response_raw**: Raw API enrichment data

### Generated Features
- Presence indicators (`has_employment_history`, `has_organization_data`)
- Length metrics (`employment_history_length`, `organization_data_length`)
- Extracted fields (`employment_history_title`, `organization_data_industry`)
- Count metrics (`employment_history_count`, `api_response_function_count`)

## 🛠️ Customization

### Modify Search Queries
```python
# Search for CEOs
df = query_instantly_api(api_key="key", search_query="CEO")

# Search by industry
df = query_instantly_api(api_key="key", search_query="tech")

# Search by location
df = query_instantly_api(api_key="key", search_query="San Francisco")
```

### Adjust Data Limits
```python
# Get more contacts
df = query_instantly_api(api_key="key", limit=5000)

# Get fewer contacts for testing
df = query_instantly_api(api_key="key", limit=100)
```

### Custom Output Filename
```python
df = query_instantly_api(
    api_key="key",
    output_file="my_contacts_export.csv"
)
```

## 🔧 Troubleshooting

### Common Issues

1. **API Key Error**:
   ```
   Error: 401 Unauthorized
   ```
   **Solution**: Check your API key in the Instantly dashboard

2. **Rate Limiting**:
   ```
   Error: 429 Too Many Requests
   ```
   **Solution**: Reduce the limit or add delays between requests

3. **Network Issues**:
   ```
   Error: Connection timeout
   ```
   **Solution**: Check your internet connection and try again

### Debug Mode

Enable detailed logging by modifying the script:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📝 Example Use Cases

### 1. Export All Contacts
```python
df = query_instantly_api(api_key="your_key")
```

### 2. Export CEO Contacts
```python
df = query_instantly_api(
    api_key="your_key",
    search_query="CEO",
    output_file="ceo_contacts.csv"
)
```

### 3. Export Tech Industry Contacts
```python
df = query_instantly_api(
    api_key="your_key",
    search_query="tech",
    output_file="tech_contacts.csv"
)
```

### 4. Export Large Dataset
```python
df = query_instantly_api(
    api_key="your_key",
    limit=10000,
    output_file="large_export.csv"
)
```

## 🔐 Security Notes

- **Never commit API keys** to version control
- **Use environment variables** for production:
  ```python
  import os
  API_KEY = os.getenv('INSTANTLY_API_KEY')
  ```
- **Rotate API keys** regularly
- **Monitor API usage** to avoid rate limits

## 📞 Support

For issues with:
- **API Access**: Contact Instantly support
- **Script Issues**: Check the troubleshooting section
- **Data Questions**: Review the output summary for column details

---

**Happy Querying! 🚀** 