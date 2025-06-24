import json

# --- Configuration ---
JSON_FILE_PATH = 'instantly_campaigns.json'
TABLE_NAME = 'leads.instantly_campaigns'
# -------------------

def generate_sql_inserts(file_path, table_name):
    """
    Reads a JSON Lines file and generates robust INSERT statements that are
    guaranteed to match the user's existing database schema by correctly
    handling all data type conversions.
    """
    # This is the schema based on your screenshot and all discovered errors.
    valid_db_schema = {
        "id": "text", "name": "text", "status": "integer", "organization": "text",
        "sequences": "jsonb", "variables": "jsonb", "timestamp": "timestamptz",
        "timestamp_created": "timestamptz", "timestamp_updated": "timestamptz",
        "campaign_id": "text", "creator_email": "text", "email_subject": "text",
        "email_body": "text", "email_body_text": "text", "email_list_name": "text",
        "email_tag_insert_unsubscribed_contact": "text", "email_gap": "integer",
        "prioritize_new_leads": "boolean",
        "prioritize_reach_before_sequences": "boolean",
        "stop_on_reply": "jsonb",
        "stop_on_auto_reply": "jsonb",
        "stop_on_company_auto_reply": "jsonb",
        "use_text_only_emails": "boolean", "text_only": "boolean",
        "link_tracking": "boolean", "open_tracking": "boolean",
        "daily_limit": "integer", "campaign_schedule": "jsonb",
        "auto_variant_select": "jsonb", "creator_name": "jsonb",
        "email_list": "jsonb", # Corrected to JSONB
        "stop_for_company": "jsonb",
        "email_tag_list": "jsonb"
    }

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                    
                    columns_to_insert = []
                    values_to_insert = []

                    for col_name, value in record.items():
                        if col_name in valid_db_schema:
                            columns_to_insert.append(f'"{col_name}"')
                            col_type = valid_db_schema[col_name]
                            
                            # --- ** THE FINAL, FINAL CORRECTED LOGIC ** ---
                            if value is None:
                                values_to_insert.append("NULL")
                            # If the database column is JSONB, always format as a valid JSON string.
                            elif col_type == 'jsonb':
                                json_string = json.dumps(value)
                                escaped_json = json_string.replace("'", "''")
                                values_to_insert.append(f"'{escaped_json}'")
                            # Handle standard types for all other columns.
                            elif isinstance(value, bool):
                                values_to_insert.append(str(value).upper())
                            elif isinstance(value, (int, float)):
                                values_to_insert.append(str(value))
                            else:
                                escaped_str = str(value).replace("'", "''")
                                values_to_insert.append(f"'{escaped_str}'")
                    
                    if columns_to_insert:
                        columns_part = ", ".join(columns_to_insert)
                        values_part = ", ".join(values_to_insert)
                        
                        sql_statement = (
                            f"INSERT INTO {table_name} ({columns_part}) VALUES ({values_part}) "
                            f"ON CONFLICT (id) DO NOTHING;"
                        )
                        print(sql_statement)

                except json.JSONDecodeError as e:
                    print(f"-- Skipping line {line_num} due to a JSON formatting error: {e}")
                    continue

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    generate_sql_inserts(JSON_FILE_PATH, TABLE_NAME)