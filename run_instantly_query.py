"""
Simple script to run Instantly API query
Usage: python run_instantly_query.py
"""

from instantly_api_query import query_instantly_api

def main():
    """Run Instantly API query with your API key."""
    
    # Replace with your actual Instantly API key
    API_KEY = "your_instantly_api_key_here"
    
    # Query parameters
    OUTPUT_FILE = "instantly_contacts_export.csv"
    LIMIT = 1000  # Number of contacts to retrieve
    SEARCH_QUERY = None  # Optional search query, e.g., "CEO" or "tech"
    
    print("🚀 Instantly API Query Tool")
    print("="*50)
    
    if API_KEY == "your_instantly_api_key_here":
        print("❌ Please update the API_KEY variable with your actual Instantly API key")
        print("   You can find your API key in your Instantly dashboard")
        return
    
    try:
        # Run the query
        df = query_instantly_api(
            api_key=API_KEY,
            output_file=OUTPUT_FILE,
            limit=LIMIT,
            search_query=SEARCH_QUERY
        )
        
        if not df.empty:
            print(f"\n✅ Success! Exported {len(df)} contacts to {OUTPUT_FILE}")
            print(f"📊 Found {len(df.columns)} total columns")
            print(f"📁 File saved in current directory")
        else:
            print("❌ No data retrieved. Please check your API key and try again.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your API key and network connection")

if __name__ == "__main__":
    main() 