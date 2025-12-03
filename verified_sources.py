import firebase_admin
from firebase_admin import credentials, firestore
import hashlib

# 1. Connect to Database
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()
REGISTRY_COLLECTION = "verified_registry" 

def _hash_url(url):
    return hashlib.md5(url.strip().lower().encode('utf-8')).hexdigest()

VERIFIED_LIST = [
    # News
    "https://www.facebook.com/bncphl",
    "https://www.facebook.com/gmanews",
    "https://www.facebook.com/abscbnNEWS",
    "https://www.facebook.com/ABSCBNnetwork",
    "https://www.facebook.com/raffytulfoinaction",
    "https://www.facebook.com/inquirerdotnet",
    "https://www.facebook.com/philstar",
    "https://www.facebook.com/rapplerdotcom",
    "https://www.facebook.com/manilabulletin",
    "https://www.facebook.com/News5Everywhere",
    "https://www.facebook.com/UNTVNewsRescue",
    "https://www.facebook.com/ptvph",
    "https://www.facebook.com/TheManilaTimes",
    "https://www.facebook.com/CNNPhilippines",
    "https://www.facebook.com/NewsWatchPlusPH",
    "https://www.facebook.com/BWorldPH",
    "https://www.facebook.com/fttmph",
    "https://www.facebook.com/visor.ph",
    "https://www.facebook.com/OneSportsPHL",
    "https://www.facebook.com/starmagicphils",
    "https://www.facebook.com/verafiles",
    "https://www.facebook.com/InteraksyonPH",
    "https://www.facebook.com/tanyagnews",
    "https://www.facebook.com/mindanews",
    "https://www.facebook.com/tribunephl",
    "https://www.facebook.com/Bulatlat.Online",
    "https://www.facebook.com/Reuters",
    "https://www.facebook.com/APNews",
    "https://www.facebook.com/AFPfra",
    "https://www.facebook.com/bloombergbusiness",
    "https://www.facebook.com/bbcnews",
    "https://www.facebook.com/bbcbreakfast",
    "https://www.facebook.com/cnninternational",
    "https://www.facebook.com/nytimes",
    "https://www.facebook.com/washingtonpost",
    "https://www.facebook.com/aljazeera",
    "https://www.facebook.com/deutschewellenews",
    "https://www.facebook.com/france24news",
    "https://www.facebook.com/NPR",
    "https://www.facebook.com/theguardian",
    "https://www.facebook.com/CNBC",
    "https://www.facebook.com/time",
    "https://www.facebook.com/USATODAY",
    "https://www.facebook.com/scmp",
    "https://www.facebook.com/nikkeiasiaofficial",
    "https://www.facebook.com/ChannelNewsAsia",
    "https://www.facebook.com/straitstimes",



    
    # Government
    "https://www.facebook.com/OfficialDOHgov",
    "https://www.facebook.com/DepartmentOfEducation.PH",
    "https://www.facebook.com/op.gov.ph",
    "https://www.facebook.com/DOSTph",
    "https://www.facebook.com/PAGASA.DOST.GOV.PH",
    "https://www.facebook.com/DOHgovPH/",
]
def seed_database():
    print(f"🚀 Seeding {len(VERIFIED_LIST)} official pages to '{REGISTRY_COLLECTION}'...")
    
    for url in VERIFIED_LIST:
        doc_id = _hash_url(url)
        
        # Data to save
        data = {
            "url": url,
            "is_verified_source": True,
            "category": "Official Entity",
            "added_at": firestore.SERVER_TIMESTAMP,
            "notes": "Manually verified by System Admin"
        }
        
        db.collection(REGISTRY_COLLECTION).document(doc_id).set(data)
        print(f"✅ Added: {url}")

    print("\n✨ Verified Database Seeding Complete!")

if __name__ == "__main__":
    seed_database()