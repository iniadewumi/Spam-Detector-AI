#%%
from downloader import all_paths
import pandas as pd


sms_spam_input_path = all_paths['sms'] / "SMSSpamCollection"
sms_df = pd.read_csv(sms_spam_input_path, delimiter='\t', header=None)
sms_df.columns = ['label', 'text']

youtube_dfs = {}
for path in all_paths['youtube-spam'].glob("*.csv"):
    df = pd.read_csv(path, delimiter='\t')
    df.rename(columns={"CLASS":"raw_label", "CONTENT":"text"}, inplace=True)
    df['source'] = str(path.name)
    youtube_dfs[str(path.name)] = df




print(df)
# %%
