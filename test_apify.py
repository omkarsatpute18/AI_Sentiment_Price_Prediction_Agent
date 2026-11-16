
import os
from dotenv import load_dotenv
from apify_client import ApifyClient

# Load environment variables
load_dotenv()

# Get Apify API token
api_token = os.getenv("APIFY_API_TOKEN")

if not api_token:
    raise ValueError("Apify API token not found in .env file!")

# Initialize Apify client
client = ApifyClient(api_token)

try:
    # Retrieve account information
    user = client.user().get()

    if user:
        print("\nApify connection successful!")
        print("Authentication: Verified")

        # Access object attributes instead of dictionary keys
        print("Account username:", getattr(user, "username", "Not available"))
        print("Account ID:", getattr(user, "id", "Not available"))

        print("\nApify API integration is working correctly!")

    else:
        print("Unable to retrieve account information.")

except Exception as error:
    print("\nApify connection failed!")
    print("Error:", error)
