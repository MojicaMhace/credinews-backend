import os
import json
from dotenv import load_dotenv

# Import functions directly from your main API file
# (Make sure this file is in the same folder as poser_detection_api.py)
from poser_detection_api import fetch_metadata, run_apify_scraper, extract_fbid, _normalize_to_page_url

# Load your .env keys
load_dotenv()

def debug_fetch(url):
    print(f"\n{'='*60}")
    print(f"🔍 DEBUGGING URL: {url}")
    print(f"{'='*60}")

    # --- PREPARE DATA ---
    base_url = _normalize_to_page_url(url)
    fbid = extract_fbid(base_url)
    print(f"ℹ️  Normalized URL: {base_url}")
    print(f"ℹ️  Extracted ID:   {fbid}")

    # --- 1. TEST META GRAPH API ---
    print(f"\n{'='*20} [1] Checking META GRAPH API {'='*20}")
    if not os.getenv("META_GRAPH_TOKEN"):
        print("❌ ERROR: META_GRAPH_TOKEN is missing in your .env file.")
    else:
        print("🚀 Sending request to Facebook Graph API...")
        graph_data = fetch_metadata(fbid)
        
        # Check for specific error flags
        if graph_data.get("error"):
            print("❌ GRAPH API ERROR:", json.dumps(graph_data, indent=2))
        elif graph_data.get("_permissions_restricted"):
            print("⚠️  GRAPH API RESTRICTED (Age/Country Gate Detected)")
            print("   (This is why you might see 'Data Unavailable' for some pages)")
        else:
            print("✅ GRAPH API SUCCESS!")
        
        print("\n--- RAW GRAPH DATA ---")
        print(json.dumps(graph_data, indent=2, default=str))

    # --- 2. TEST APIFY SCRAPER ---
    print(f"\n{'='*20} [2] Checking APIFY SCRAPER {'='*20}")
    if not os.getenv("APIFY_TOKEN"):
        print("❌ ERROR: APIFY_TOKEN is missing in your .env file.")
    else:
        print("🚀 Starting Apify Task (This will take 10-30 seconds)...")
        print("   (If this fails, check your Apify usage limits or Token)")
        
        try:
            apify_data = run_apify_scraper(base_url)
            
            if not apify_data:
                print("❌ APIFY RETURNED NONE (Failed to scrape or blocked)")
            else:
                print("✅ APIFY SUCCESS!")
                print("\n--- RAW APIFY DATA ---")
                print(json.dumps(apify_data, indent=2, default=str))
        except Exception as e:
            print(f"❌ APIFY CRASHED: {e}")

if __name__ == "__main__":
    # Ask for URL in the terminal
    target_url = input("\nPaste the Facebook Page URL to test: ").strip()
    if target_url:
        debug_fetch(target_url)