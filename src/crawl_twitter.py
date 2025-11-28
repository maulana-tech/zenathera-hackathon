import subprocess
import pandas as pd
import os

# Configuration
# NOTE: You must provide a valid Twitter Auth Token here.
TWITTER_AUTH_TOKEN = '721673238267263t71t361t361777' 
FILENAME = 'Coretax.csv'
SEARCH_KEYWORD = 'CoreTax since:2024-01-01 until:2025-11-20 lang:id'
LIMIT = 1000

# Determine paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_FILE = os.path.join(DATA_DIR, FILENAME)

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

print(f"Crawling data for keyword: {SEARCH_KEYWORD}")
print(f"Output file: {OUTPUT_FILE}")

# Construct command
# npx -y tweet-harvest@2.6.1 -o "{filename}" -s "{search_keyword}" --tab "LATEST" -l {limit} --token {twitter_auth_token}
command = [
    "npx",
    "-y",
    "tweet-harvest@2.6.1",
    "-o", FILENAME,
    "-s", SEARCH_KEYWORD,
    "--tab", "LATEST",
    "-l", str(LIMIT),
    "--token", TWITTER_AUTH_TOKEN
]

# Run command
try:
    print("Starting tweet-harvest...")
    # We execute the command in the DATA_DIR so the output file is created there
    subprocess.run(command, cwd=DATA_DIR, check=True)
    print("Crawling completed successfully.")
    
    # Read and display
    if os.path.exists(OUTPUT_FILE):
        df = pd.read_csv(OUTPUT_FILE)
        print(f"Data loaded successfully.")
        print(f"Total tweets: {len(df)}")
        print(df.head())
    else:
        print(f"Warning: Output file not found at {OUTPUT_FILE}")

except subprocess.CalledProcessError as e:
    print(f"Error occurred while running tweet-harvest: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
