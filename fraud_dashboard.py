import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Data Function
def load_data():
    df = pd.read_csv("PS_20174392719_1491204439457_log.csv")  # Use your correct file path here
    return df

df = load_data()

# Title of the Dashboard
st.title("UPI Fraud Detection Dashboard")

# Display Dataset
if st.checkbox('Show raw data'):
    st.subheader('Raw Dataset')
    st.write(df.head())

# Fraudulent Transactions by Type (Bar Chart)
st.subheader("Number of Fraudulent Transactions by Transaction Type")
fraud_by_type = df.groupby('type')['isFraud'].sum().reset_index()

fig, ax = plt.subplots()
sns.barplot(x='type', y='isFraud', data=fraud_by_type, palette='Blues_d', ax=ax)
ax.set_xlabel('Transaction Type')
ax.set_ylabel('Number of Fraudulent Transactions')
ax.set_title('Fraudulent Transactions by Type')
st.pyplot(fig)

# Fraud Percentage by Transaction Type
st.subheader("Percentage of Fraudulent Transactions by Transaction Type")
fraud_percentage = df.groupby('type')['isFraud'].mean() * 100

fig2, ax2 = plt.subplots()
sns.barplot(x=fraud_percentage.index, y=fraud_percentage.values, palette='coolwarm', ax=ax2)
ax2.set_xlabel('Transaction Type')
ax2.set_ylabel('Percentage of Fraudulent Transactions (%)')
ax2.set_title('Percentage of Fraudulent Transactions by Type')
st.pyplot(fig2)

# Transaction Amount Distribution (Histogram)
st.subheader("Transaction Amount Distribution")
fig3, ax3 = plt.subplots()
sns.histplot(df['amount'], bins=100, kde=True, color='green', ax=ax3)
ax3.set_xscale('log')
ax3.set_xlabel('Transaction Amount (Log Scale)')
ax3.set_ylabel('Frequency')
ax3.set_title('Transaction Amount Distribution')
st.pyplot(fig3)

# Fraud Over Time (Line Plot)
st.subheader("Fraudulent Transactions Over Time (Step)")
fraud_by_step = df.groupby('step')['isFraud'].sum().reset_index()

fig4, ax4 = plt.subplots()
sns.lineplot(x='step', y='isFraud', data=fraud_by_step, color='red', ax=ax4)
ax4.set_xlabel('Step (Time)')
ax4.set_ylabel('Number of Fraudulent Transactions')
ax4.set_title('Fraudulent Transactions Over Time')
st.pyplot(fig4)

# Correlation Heatmap (for all numeric features)
st.subheader("Correlation Heatmap of Features")

# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# If numeric_df is empty (i.e., no numeric columns), skip the heatmap
if not numeric_df.empty:
    fig5, ax5 = plt.subplots(figsize=(10,6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax5)
    ax5.set_title('Correlation Matrix of Numeric Features')
    st.pyplot(fig5)
else:
    st.write("No numeric columns found for correlation heatmap.")

# Filter by Transaction Type
st.sidebar.subheader("Filter Data by Transaction Type")
transaction_types = df['type'].dropna().unique()  # Drop NaN values and get unique transaction types
selected_type = st.sidebar.selectbox("Select a Transaction Type", transaction_types)

# Filter the data based on selected transaction type
filtered_df = df[df['type'] == selected_type]

# Display filtered data
st.subheader(f"Filtered Data for Transaction Type: {selected_type}")
st.write(filtered_df.head())

# Display Fraud and Non-Fraud Distribution for Selected Type
st.subheader(f"Fraud Distribution for {selected_type}")
fraud_dist = filtered_df['isFraud'].value_counts(normalize=True) * 100
fig6, ax6 = plt.subplots()
sns.barplot(x=fraud_dist.index, y=fraud_dist.values, ax=ax6, palette='Set1')
ax6.set_xticklabels(['Non-Fraud', 'Fraud'])
ax6.set_ylabel('Percentage (%)')
ax6.set_title(f'Fraud Distribution for {selected_type}')
st.pyplot(fig6)
