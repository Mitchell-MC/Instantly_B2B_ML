import base64

# Your API key
api_key = "ZjQ0Mjc3ZGUtNjliMi00YmMzLWE2OWMtMjhhZmQ0MDk0MTIzOkx5VWZ6UnB6RmR3Zw=="

try:
    # Decode the base64 string
    decoded = base64.b64decode(api_key).decode('utf-8')
    print(f"Decoded API key: {decoded}")
    
    # Split by colon if it's in format key:secret
    if ':' in decoded:
        parts = decoded.split(':')
        print(f"Key part: {parts[0]}")
        print(f"Secret part: {parts[1]}")
    
except Exception as e:
    print(f"Error decoding: {e}") 