import pandas as pd
df = pd.read_parquet("out/ETHUSDT_1m.parquet").sort_values("ts")
print(len(df), df.iloc[0]["ts"], "→", df.iloc[-1]["ts"])
print((df["ts"].diff()>pd.Timedelta("1min")).sum(), "gaps>1min")
print(df.head(3))
print(df.tail(3))
