import time
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
start_apriori = time.perf_counter() 
freq_apriori = apriori(df, min_support, use_colnames = True)
end_apriori = time.perf_counter()

print('\nFrequent itemsets (Apriori):')
print(freq_apriori)
print(f'\nExecution Time: {(end_apriori - start_apriori) * 1000} ms')

# FP-Growth
start_fpgrowth = time.perf_counter() 
freq_fpgrowth = fpgrowth(df, min_support, use_colnames = True)
end_fpgrowth = time.perf_counter()

print('\nFrequent itemsets (FP-Growth):')
print(freq_fpgrowth)
print(f'\nExecution Time: {(end_fpgrowth - start_fpgrowth) * 1000} ms')
