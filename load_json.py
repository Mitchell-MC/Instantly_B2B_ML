import json

# --- Configuration ---
JSON_FILE_PATH = 'instantly_campaigns.json'
SQL_OUTPUT_FILE = 'inserts.sql'

CAMPAIGNS_TABLE = 'leads.instantly_campaigns'
STEPS_TABLE = 'leads.instantly_campaign_steps'
VARIANTS_TABLE = 'leads.instantly_campaign_contents'
# -------------------

def format_sql_value(value):
    """Safely formats a Python value for an SQL INSERT statement."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return str(value).upper()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (dict, list)):
        json_string = json.dumps(value)
        escaped_json = json_string.replace("'", "''")
        return f"'{escaped_json}'"
    escaped_str = str(value).replace("'", "''")
    return f"'{escaped_str}'"


def generate_relational_sql_inserts(file_path, output_file):
    """
    Reads a JSON Lines file and writes SQL INSERT statements for a normalized,
    relational database schema using composite keys.
    """
    campaign_schema = {
        "id", "name", "status", "organization", "timestamp_created",
        "timestamp_updated", "daily_limit", "stop_on_reply", "link_tracking",
        "open_tracking", "stop_on_auto_reply", "prioritize_new_leads",
        "stop_for_company", "insert_unsubscribe_header", "email_gap",
        "campaign_schedule", "email_list", "text_only", "email_tag_list",
        "auto_variant_select"
    }

    with open(output_file, 'w', encoding='utf-8') as outfile:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue

                    try:
                        record = json.loads(line)
                        campaign_id = record.get('id')
                        if not campaign_id:
                            outfile.write(f"-- Skipping line {line_num} because it has no 'id' field.\n")
                            continue

                        # Wrap all inserts for one campaign in a single transaction
                        outfile.write("BEGIN;\n")

                        # 1. INSERT for Campaigns Table (no changes here)
                        campaign_cols = []
                        campaign_vals = []
                        for col_name, value in record.items():
                            if col_name in campaign_schema:
                                campaign_cols.append(f'"{col_name}"')
                                campaign_vals.append(format_sql_value(value))
                        
                        if campaign_cols:
                            cols_part = ", ".join(campaign_cols)
                            vals_part = ", ".join(campaign_vals)
                            sql_campaign = (
                                f"INSERT INTO {CAMPAIGNS_TABLE} ({cols_part}) VALUES ({vals_part}) "
                                f"ON CONFLICT (id) DO NOTHING;\n"
                            )
                            outfile.write(sql_campaign)

                        # 2. INSERTs for Steps and Variants
                        sequences = record.get('sequences', [])
                        if isinstance(sequences, list) and sequences:
                            steps = sequences[0].get('steps', [])
                            for step_index, step in enumerate(steps):
                                step_type = format_sql_value(step.get('type'))
                                step_delay = format_sql_value(step.get('delay'))
                                campaign_id_sql = format_sql_value(campaign_id)

                                # Script Change: No longer has an 'id' column to insert
                                sql_step = (
                                    f"INSERT INTO {STEPS_TABLE} (campaign_id, step_index, type, delay) "
                                    f"VALUES ({campaign_id_sql}, {step_index}, {step_type}, {step_delay}) "
                                    f"ON CONFLICT (campaign_id, step_index) DO NOTHING;\n"
                                )
                                outfile.write(sql_step)

                                variants = step.get('variants', [])
                                for variant_index, variant in enumerate(variants):
                                    if variant.get("v_disabled") is True:
                                        continue
                                    
                                    subject = format_sql_value(variant.get('subject'))
                                    body = format_sql_value(variant.get('body'))
                                    
                                    # Script Change: The variant insert is now much simpler.
                                    # It no longer needs a sub-select. It directly inserts the values.
                                    sql_variant = (
                                        f"INSERT INTO {VARIANTS_TABLE} (campaign_id, step_index, variant_index, subject, body) "
                                        f"VALUES ({campaign_id_sql}, {step_index}, {variant_index}, {subject}, {body}) "
                                        f"ON CONFLICT (campaign_id, step_index, variant_index) DO NOTHING;\n"
                                    )
                                    outfile.write(sql_variant)
                        
                        outfile.write("COMMIT;\n-- End of record --\n\n")

                    except json.JSONDecodeError as e:
                        outfile.write(f"ROLLBACK;\n-- Skipping line {line_num} due to a JSON formatting error: {e}\n\n")
                        continue

        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    generate_relational_sql_inserts(JSON_FILE_PATH, SQL_OUTPUT_FILE)
    print(f"Successfully created the SQL file: {SQL_OUTPUT_FILE}")