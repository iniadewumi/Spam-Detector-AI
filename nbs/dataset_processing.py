from downloader import all_paths
import pandas as pd

SPAM_DATASET_PATH = all_paths['exports'] / "spam-dataset.csv"
sms_spam_input_path = all_paths['sms'] / "SMSSpamCollection"
sms_df = pd.read_csv(sms_spam_input_path, delimiter='\t', header=None)
sms_df.columns = ['label', 'text']
sms_df['source'] = "sms_spam"
youtube_dfs = {}
for path in all_paths['youtube-spam'].glob("*.csv"):
    df = pd.read_csv(path)
    df.rename(columns={"CLASS":"raw_label", "CONTENT":"text"}, inplace=True)
    df['label'] = df['raw_label'].apply(lambda x: "spam" if str(x)=="1" else "ham")
    df['source'] = str(path.name)
    youtube_dfs[str(path.name)] = df

full = list(youtube_dfs.values())
full.append(sms_df)
full_df = pd.concat(full)[['label', 'text', 'source']]


full_df.to_csv(SPAM_DATASET_PATH, index=False)

print(youtube_dfs)

