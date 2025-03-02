from utils.text_cleaning import cleanText

def process_text(df):
    df['news'] = df['news'].apply(cleanText)
    return df
