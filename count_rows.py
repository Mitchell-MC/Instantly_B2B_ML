import pandas as pd

def count_specific_rows(filename):
    """
    Counts rows in a CSV based on several conditions, including the
    presence or absence of an email body.

    Args:
        filename (str): The path to the CSV file to analyze.
    """
    print(f"Analyzing file: {filename}")
    try:
        # Load the CSV file into a pandas DataFrame, using 'latin1' encoding
        df = pd.read_csv(filename, encoding='latin1')

        # --- Define the Conditions for Filtering ---

        # Condition 1: 'upload_method' column contains the substring 'api'.
        # 'na=False' ensures that any blank cells in this column are not counted.
        condition1 = df['upload_method'].str.contains('api', na=False)

        # Condition 2: 'email_bodies' column is not null/empty.
        # .notna() is a reliable way to check for any non-empty value.
        condition2 = df['email_bodies'].notna()
        
        # --- NEW: Condition for rows WITHOUT an email body ---
        # .isna() checks for null/empty values.
        condition_no_email = df['email_bodies'].isna()


        # --- Apply the Filters and Count ---

        # Combine the original two conditions using the '&' operator
        filtered_df = df[condition1 & condition2]

        # The count is the number of rows in the filtered DataFrame
        count = len(filtered_df)
        
        # --- NEW: Get the count of rows with no email body ---
        count_without_email = condition_no_email.sum()


        # --- Updated Output Section ---
        print("-" * 30)
        print(f"Total rows in file: {len(df)}")
        print(f"Rows with 'upload_method' containing 'api': {condition1.sum()}")
        print(f"Rows with non-empty 'email_bodies': {condition2.sum()}")
        print(f"Rows with empty 'email_bodies': {count_without_email}") # <-- ADDED THIS LINE
        print("-" * 30)
        print(f"Final Count: Found {count} rows that meet BOTH 'api' AND have a non-empty email body.")
        print("-" * 30)


    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found. Please make sure it's in the same directory as the script.")
    except KeyError as e:
        print(f"Error: A required column was not found: {e}. Please check your CSV file's headers.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Set the name of the file you want to analyze.
    # This should be the output from the previous merge script.
    target_file = 'merged_contacts.csv'

    # Run the analysis function
    count_specific_rows(target_file)