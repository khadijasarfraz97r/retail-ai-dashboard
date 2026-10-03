import pandas as pd

# Load dataset
data = pd.read_excel("Online Retail.xlsx")

# Show first 5 rows
print("First 5 rows:")
print(data.head())

# Dataset size
print("\nDataset Shape:")
print(data.shape)

# Column information
print("\nDataset Information:")
print(data.info())

# Missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Duplicate rows
print("\nDuplicate Rows:")
print(data.duplicated().sum())
# ==========================================
# 1. REMOVE MISSING CUSTOMER IDs
# ==========================================

data = data.dropna(subset=["CustomerID"])

print("\nAfter removing missing Customer IDs:")
print(data.shape)


# ==========================================
# 2. REMOVE CANCELLED ORDERS
# ==========================================

data = data[~data["InvoiceNo"].astype(str).str.startswith("C")]

print("\nAfter removing cancelled orders:")
print(data.shape)


# ==========================================
# 3. REMOVE INVALID VALUES
# ==========================================

data = data[data["Quantity"] > 0]
data = data[data["UnitPrice"] > 0]


# ==========================================
# 4. CREATE TOTAL SPENDING
# ==========================================

data["TotalPrice"] = data["Quantity"] * data["UnitPrice"]


# ==========================================
# 5. CONVERT DATE
# ==========================================

data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])


# ==========================================
# 6. CREATE REFERENCE DATE
# ==========================================

reference_date = data["InvoiceDate"].max() + pd.Timedelta(days=1)


# ==========================================
# 7. RFM ANALYSIS
# ==========================================

rfm = data.groupby("CustomerID").agg({

    "InvoiceDate": lambda x:
        (reference_date - x.max()).days,

    "InvoiceNo": "nunique",

    "TotalPrice": "sum"
})


# Rename columns

rfm.columns = [
    "Recency",
    "Frequency",
    "Monetary"
]


# ==========================================
# 8. SHOW RFM DATA
# ==========================================

print("\nRFM Analysis:")
print(rfm.head())

print("\nNumber of Customers:")
print(len(rfm))


# ==========================================
# 9. SAVE RFM DATA
# ==========================================

rfm.to_csv("customer_rfm.csv")

print("\nRFM data saved successfully!")