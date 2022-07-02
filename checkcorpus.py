from tqdm import tqdm
import pandas as pd

final_df = pd.read_csv('sec_dataset_III.csv')

words, errors, etypes = [], [], []
for idx in tqdm(range(len(final_df))):
    word = final_df.iloc[idx, 0]
    error = final_df.iloc[idx, 1]
    etype = final_df.iloc[idx, 2]

    if word != error:
        if error not in errors:
            words.append(word)
            errors.append(error)
            etypes.append(etype)

final_df_2 = pd.DataFrame({
    'Word': words,
    'Error': errors,
    'ErrorType': etypes
})

print(len(final_df), len(final_df_2))
if len(final_df_2) < len(final_df):
    final_df_2.to_csv('sec_dataset_III_v2.csv', index=False)
    print('sec_dataset_III_v2.csv has been saved')

if __name__ == '__main__':
    pass
