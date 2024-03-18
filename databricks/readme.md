![image](https://github.com/SHRIDHARKN/data_science/assets/74343939/aa710bbe-b961-4d7c-9476-24097ea91aec)

### convert date col in string to required format
`
data = [("15-01-2023",), ("20-02-2023",), ("25-03-2023",)]
df = spark.createDataFrame(data, ["date_str"])
df = df.withColumn("date", to_date(df["date_str"], "dd-MM-yyyy"))
`
