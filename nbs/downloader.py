import subprocess
import pathlib, os
import zipfile



BASE_DIR = pathlib.Path().resolve().parent / "AI-API"
DATASETS_DIR = BASE_DIR / "datasets"
ZIPS_DIR = DATASETS_DIR / "zips"
SPAM_CLASSIFIER_DIR = DATASETS_DIR / "spam-classifier"
SMS_SPAM_DIR = SPAM_CLASSIFIER_DIR / "sms-spam" 
YOUTUBE_SPAM_DIR = SPAM_CLASSIFIER_DIR / "youtube-spam"
EXPORTS_DIR = DATASETS_DIR / "exports"
SPAM_DATASET_PATH = EXPORTS_DIR / "spam-dataset.csv"
METADATA_PATH = EXPORTS_DIR / "spam-metadata.pkl"
TOKENIZER_PATH = EXPORTS_DIR / "spam-tokenizer.json"


all_paths = {'base': BASE_DIR, 'datasets': DATASETS_DIR, 'zips': ZIPS_DIR, 
            'spam-classifier': SPAM_CLASSIFIER_DIR, 'sms': SMS_SPAM_DIR, 
            'youtube-spam': YOUTUBE_SPAM_DIR, "exports": EXPORTS_DIR, 
            "full_spam_df": SPAM_DATASET_PATH, "metadata": METADATA_PATH, "tokenizer":TOKENIZER_PATH}



def main():
    os.makedirs(ZIPS_DIR, exist_ok=True)        
    os.makedirs(SMS_SPAM_DIR, exist_ok=True)
    os.makedirs(YOUTUBE_SPAM_DIR, exist_ok=True)
    os.makedirs(EXPORTS_DIR, exist_ok=True)

    SMS_ZIP = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    YOUTUBE_ZIP = "https://archive.ics.uci.edu/ml/machine-learning-databases/00380/YouTube-Spam-Collection-v1.zip"


    def download_spam_collection(link, filename):
        if not os.path.exists(f'{ZIPS_DIR}/{filename}'):
            subprocess.run(["curl", f"{link}", "-o", f'{ZIPS_DIR}/{filename}'])
        else:
            print(f"File {filename} exist!")

    download_spam_collection(YOUTUBE_ZIP, filename="youtube-spam-dataset.zip")
    download_spam_collection(SMS_ZIP, filename="spam-dataset.zip")



    def unzip_spam_collection(filepath, destination):
        if len(os.listdir(destination))<1:
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(destination)
                return
        loc = str(destination).split('\\')[-1]
        print(f"Did not unzip because {loc} already contains files")
        
    unzip_spam_collection(filepath=ZIPS_DIR/"spam-dataset.zip", destination=SMS_SPAM_DIR)
    unzip_spam_collection(filepath=ZIPS_DIR/"youtube-spam-dataset.zip", destination=YOUTUBE_SPAM_DIR)

if __name__ == '__main__':
    main()