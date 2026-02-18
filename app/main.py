import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth

# Import initial data
df = pd.read_csv('data.csv')

# One-hot encode dataset
transaction_groups = df.groupby('Transaction')['Item'].apply(list).reset_index()
data = transaction_groups['Item'].tolist()
te = TransactionEncoder()
te_ary = te.fit(data).transform(data)
df = pd.DataFrame(te_ary, columns = te.columns_)

# rroughly 5.28%
min_support = 500 / len(df)

# Apriori
freq_apriori = apriori(df, min_support, use_colnames = True)
print('\nFrequent itemsets (Apriori):')
print(freq_apriori)

# FP-Growth
freq_fpgrowth = fpgrowth(df, min_support, use_colnames = True)
print('\nFrequent itemsets (Apriori):')
print(freq_fpgrowth)