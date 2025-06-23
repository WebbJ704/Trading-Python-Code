import os 
from dotenv import load_dotenv

load_dotenv()

api = os.getenv("QUANDL_API_KEY")
if api:
    print(f"API Key Loaded: {api}")
else:
    print("API Key not found. Make sure it is set correctly.")

