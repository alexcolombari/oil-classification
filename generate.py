import pandas as pd

# Generate Pickle archive

data_folder = "/opt/data_repository/oil_samples/"
file_to_open = data_folder + "laminas.pkl"
df = pd.read_pickle(file_to_open)

aa = df.iloc[4000, :]
aa.to_pickle("/opt/data_repository/oil_samples/half-samples.pkl")
