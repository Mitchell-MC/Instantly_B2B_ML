import pandas as pd

def merge_csv_files(enriched_contacts_file, campaigns_file, output_file):
    """
    Performs a many-to-one merge between two CSV files.

    Args:
        enriched_contacts_file (str): The path to the enriched_contacts.csv file.
        campaigns_file (str): The path to the instantly_campaigns_with_emails.csv file.
        output_file (str): The path to save the merged CSV file.
    """
    try:
        # Load the two CSV files into pandas DataFrames
        print("Loading files with latin1 encoding...")
        # Added encoding='latin1' to handle potential encoding issues
        enriched_contacts_df = pd.read_csv(enriched_contacts_file, encoding='latin1')
        campaigns_df = pd.read_csv(campaigns_file, encoding='latin1')

        # --- Data Cleaning (Optional but Recommended) ---
        # Rename the 'id' column in the campaigns dataframe to avoid confusion after merging
        campaigns_df.rename(columns={'id': 'campaign_id'}, inplace=True)

        print("Performing the merge...")
        # Perform the many-to-one merge.
        # We use a 'left' merge to ensure all records from the 'enriched_contacts' file are kept.
        merged_df = pd.merge(
            enriched_contacts_df,
            campaigns_df,
            left_on='campaign',
            right_on='campaign_id',
            how='left'
        )

        # --- Post-Merge Cleanup (Optional) ---
        # You might want to drop the redundant campaign_id column
        # merged_df.drop('campaign_id', axis=1, inplace=True)


        # Save the merged DataFrame to a new CSV file
        print(f"Saving merged file to {output_file}...")
        merged_df.to_csv(output_file, index=False)

        print("Merge complete! The new file is ready.")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please make sure the CSV files are in the correct directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Define the names of your input and output files
    enriched_contacts_filename = 'enriched_contacts.csv'
    campaigns_filename = 'instantly_campaigns_with_emails.csv'
    output_filename = 'merged_contacts.csv'

    # Run the merge function
    merge_csv_files(enriched_contacts_filename, campaigns_filename, output_filename)