import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Ignore FutureWarnings - very annoying when trying to view printouts. Had to adjust code here so those warning go away
# One suggestion was update Seaborn, Already using most current version.**
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Read in the csv for data cleaning
df = pd.read_csv('churn_raw_data.csv')
print(df.head())


churn_rate = df['Churn'].value_counts(normalize=True)['Yes'] * 100
print(f"Churn Rate: {churn_rate:.2f}%")

# Print out the data frames types(Int, Object, Float)
print(df.dtypes)

# Group columns by their dtype
dtype_groups = df.columns.groupby(df.dtypes)

# Print out each data type and its columns of each data type
for dtype, columns in dtype_groups.items():
    print(f"\nData Type: {dtype}")
    for column in columns:
        print(f"- {column}")

# View the columns in the data frame
cols = df.columns
print(cols)

################################################################
# Group columns by category
customer_info_columns = ['CaseOrder', 'Customer_id', 'Interaction']
customer_account_columns = ['Outage_sec_perweek', 'Email', 'Contacts', 'Yearly_equip_failure', 'Techie', 'Contract',
                            'Port_modem', 'Tablet', 'InternetService', 'Phone', 'Multiple', 'OnlineSecurity',
                            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                            'PaperlessBilling', 'PaymentMethod', 'Tenure', 'Bandwidth_GB_Year', 'MonthlyCharge',
                            'Churn']
demographics_columns = ['City', 'State', 'County', 'Zip', 'Lat', 'Lng', 'Population', 'Area', 'Timezone', 'Job',
                        'Children', 'Age', 'Education', 'Employment', 'Income', 'Marital', 'Gender']
questionnaire_columns = ['item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'item7', 'item8']

# Get counts for each group
count_customer_info = len(customer_info_columns)
count_customer_account = len(customer_account_columns)
count_demographics = len(demographics_columns)
count_questionnaire = len(questionnaire_columns)

# Calculate the total count from the groups
total_group_count = count_customer_info + count_customer_account + count_demographics + count_questionnaire

# Get the total column count from the DataFrame
total_df_count = len(df.columns)

# Check if they match
if total_group_count == total_df_count:
    print("All columns have been accounted for!")
else:
    print(f"Missed {total_df_count - total_group_count} columns.")


# Combine all your groups into one list
all_grouped_columns = customer_info_columns + customer_account_columns + demographics_columns + questionnaire_columns

# Convert the lists to sets and find the difference
missed_columns = set(df.columns) - set(all_grouped_columns)

# Check if any columns were missed
if missed_columns:
    print(f"You missed the following columns: {', '.join(missed_columns)}")
else:
    print("All columns have been accounted for!")

##################################################################
# Print duplicated rows
print(df[df.duplicated()])
# Remove the Unnamed column, don't need two columns counting or indexing
df = df.drop(df.columns[0], axis=1)
print(df.head())

#################################################################
# Get count of missing values
missing_values_count = df.isna().sum()
missing_values_count = missing_values_count[missing_values_count > 0]
print(missing_values_count)


# Fills in null values, Using Imputation with either Mean, Median or Mode - Depending on the category and data type
df['Children'].fillna(df['Children'].median(), inplace=True)
df['Income'].fillna(df['Income'].median(), inplace=True)
df['Bandwidth_GB_Year'].fillna(df['Bandwidth_GB_Year'].median(), inplace=True)
df['Tenure'].fillna(df['Tenure'].median(), inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)

df['Techie'].fillna(df['Techie'].mode()[0], inplace=True)
df['Phone'].fillna(df['Phone'].mode()[0], inplace=True)
df['TechSupport'].fillna(df['TechSupport'].mode()[0], inplace=True)
df['InternetService'].fillna(df['InternetService'].mode()[0], inplace=True)


if df.isna().any().any():
    print("There are missing values.")
else:
    print("No missing values.")
##############################################################################


# Loop through each group and its columns to view unique values
def print_unique_values(group_name, columns):
    print(f"\n{group_name}:\n" + "="*40)
    for col in columns:
        unique_vals = df[col].unique()
        print(f"{col} Unique Values:\n{unique_vals}\n")


# Customer Info Group
print_unique_values("Customer Info", customer_info_columns)

# Customer Account Group
print_unique_values("Customer Account", customer_account_columns)

# Demographics Group
print_unique_values("Demographics", demographics_columns)

# Questionnaire Group
print_unique_values("Questionnaire", questionnaire_columns)

#####################################################################
#####################################################################
# Another way to print unique values by each data type - Not included in paper

# int_cols = df.select_dtypes(include=['int64']).columns.tolist()
# float_cols = df.select_dtypes(include=['float64']).columns.tolist()
# object_cols = df.select_dtypes(include=['object']).columns.tolist()
#
# for col in object_cols:
#     print(f"{col}: Unique values = {df[col].unique()}")

#####################################################################
#####################################################################
# Relabel the columns listed as item1...item8 with appropriate questions
df.rename(columns={'item1': 'Timely response',
                   'item2': 'Timely fixes',
                   'item3': 'Timely replacement',
                   'item4': 'Reliability',
                   'item5': 'Options',
                   'item6': 'Respectful response',
                   'item7': 'Courteous exchange',
                   'item8': 'Evidence of active listening'},
          inplace=True)
################################################################
numeric_std = df.select_dtypes(include=['float64', 'int64']).std()
print(numeric_std)

################################################################
# Extract Values, Create Bins and Labels for the values that I want grouped.
# Grouping these values will allow easier views to evaluate groups of customers who have similar values,
# removes the decimal and helps group things a little better to deal with outliers
# make a copy df for handling my newly created value groups, for easier viewing
df_group = df.copy()
tenure_values = df['Tenure'].values
tenure_bins = [0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
tenure_labels = ['0-5', '5-10', '10-15', '15-20', '20-30', '30-40', '40-50',
                 '50-60', '60-70', '70-80', '80-90', '90-100']
df_group['tenure_value_group'] = pd.cut(tenure_values, bins=tenure_bins, labels=tenure_labels, right=False)
df['Tenure'] = df['Tenure'].apply(lambda x: int(x))
# print(df['tenure_value_group'].unique())

gb_year_values = df['Bandwidth_GB_Year'].values
gb_bins = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 10000]
gb_labels = ['0-500', '500-1000', '1000-1500', '1500-2000', '2000-2500', '2500-3000', '3000-3500', '3500-4000',
             '4000-4500', '4500-5000', '5000-5500', '5500-10000']
df_group['gb_value_group'] = pd.cut(gb_year_values, bins=gb_bins, labels=gb_labels, right=False)
df['Bandwidth_GB_Year'] = df['Bandwidth_GB_Year'].apply(lambda x: int(x))
# print(df['gb_value_group'].unique())

income_values = df['Income'].values
income_bins = [0, 25000, 40000, 60000, 80000, 100000, 200000]
income_labels = ['0-25000', '25000-40000', '40000-60000', '60000-80000', '80000-100000', '100000-250000']
df_group['income_value_group'] = pd.cut(income_values, bins=income_bins, labels=income_labels, right=False)
df['Income'] = df['Income'].apply(lambda x: int(x))
# print(df['income_value_group'].unique())

outage_values = df['Outage_sec_perweek'].values
outage_bins = [0, 1, 2, 3, 4, 5, 10, 30, 60, 120]
outage_labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-10', '10-30', '30-60', '60-120']
df_group['outage_value_group'] = pd.cut(outage_values, bins=outage_bins, labels=outage_labels, right=False)
df['Outage_sec_perweek'] = df['Outage_sec_perweek'].apply(lambda x: int(x))
# print(df['outage_value_group'].unique())

monthly_charge_values = df['MonthlyCharge'].values
monthly_bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 500, 1000]
monthly_labels = ['0-50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-350', '350-400', '400-500',
                  '500-1000']
df_group['monthly_charge_value_group'] = pd.cut(monthly_charge_values, bins=monthly_bins, labels=monthly_labels, right=False)
df['MonthlyCharge'] = df['MonthlyCharge'].apply(lambda x: int(x))
# print(df['monthly_charge_value_group'].unique())

children_values = df['Children'].values
children_bins = [0, 1, 2, 3, 4, 5, 6, 10]
children_labels = ['0', '1', '2', '3', '4', '5', '6-10']
df_group['children_value_group'] = pd.cut(children_values, bins=children_bins, labels=children_labels, right=False)
df['Children'] = df['Children'].apply(lambda x: int(x))
# print(df['children_value_group'].unique())

columns_to_plot = ['tenure_value_group', 'gb_value_group', 'income_value_group',
                   'outage_value_group', 'monthly_charge_value_group', 'children_value_group']

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

for idx, col in enumerate(columns_to_plot):
    row = idx // 3
    col_idx = idx % 3
    df_group[col].value_counts().sort_index().plot(kind='bar', ax=axes[row, col_idx])
    axes[row, col_idx].set_title(col)
    axes[row, col_idx].set_ylabel('Count')

plt.tight_layout()
plt.savefig('all_binned_data_bar_plots.jpg')
plt.show()

print(df.head())


# Create a Histogram, so I can view my chosen values or groups.
sns.histplot(df_group['outage_value_group'], bins=30, kde=True)
plt.title('Distribution of Outage_sec_perweek Column')
plt.show()

sns.histplot(df_group['gb_value_group'], bins=30, kde=True)
plt.title('Distribution of Bandwidth_GB_Year Column')
plt.show()

sns.histplot(df_group['monthly_charge_value_group'], bins=30, kde=True)
plt.title('Distribution of MonthlyCharge Column')
plt.show()

sns.histplot(df_group['tenure_value_group'], bins=30, kde=True)
plt.title('Distribution of Tenure Column')
plt.show()

sns.histplot(df_group['income_value_group'], bins=30, kde=True)
plt.title('Distribution of Income Column')
plt.show()

sns.histplot(df_group['children_value_group'], bins=30, kde=True)
plt.title('Distribution of Children Column')
plt.show()

# Plot histograms with custom size
ax = df[['Children', 'Age', 'Income', 'Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year']].hist(grid=False, bins=20,
                                                                                            rwidth=0.9, color='#86bf91',
                                                                                            zorder=2, figsize=(18, 10))

# Provide a central title for all subplots
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('histogram.jpg')
plt.show()


################################################################
columns_to_boxplot = ['Age', 'Tenure', 'Bandwidth_GB_Year']

# Box Plotting
plt.figure(figsize=(15, 7))
for idx, col in enumerate(columns_to_boxplot):
    plt.subplot(1, 3, idx + 1)
    df[col].plot(kind='box', vert=True)
    plt.title(col)

plt.tight_layout()
plt.savefig('boxplots.jpg')
plt.show()

more_columns_to_plot = ['Income', 'MonthlyCharge', 'Children']
# Box Plotting
plt.figure(figsize=(15, 7))
for idx, col in enumerate(more_columns_to_plot):
    plt.subplot(1, 3, idx + 1)
    df[col].plot(kind='box', vert=True)
    plt.title(col)

plt.tight_layout()
plt.savefig('boxplots2.jpg')
plt.show()

# ################################################################
# Select columns that have relevance to the churn rate
selected_data = ['Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year', 'Timely response', 'Timely fixes',
                 'Timely replacement', 'Reliability', 'Options', 'Courteous exchange', 'Respectful response',
                 'Evidence of active listening']

# Select the data from your dataframe
data_subset = df[selected_data]
data_subset = data_subset.select_dtypes(include=[np.number])

# Standardize the data
scaled_data = StandardScaler().fit_transform(data_subset)

# Perform PCA
pca = PCA()
pca.fit(scaled_data)

# Total number of principal components
total_components = pca.n_components_
print(f"Total number of principal components: {total_components}")

# Create DataFrame for loading matrix
loading_matrix_df = pd.DataFrame(pca.components_, columns=selected_data)

# Naming the indices as PC1, PC2, etc.
loading_matrix_df.index = [f"PC{i+1}" for i in range(loading_matrix_df.shape[0])]
loading_matrix_df.to_csv('loading_matrix.csv')
print(loading_matrix_df)


# Get explained variance
explained_variance = pca.explained_variance_ratio_

# Extract eigenvalues
eigenvalues = pca.explained_variance_

print(eigenvalues)

loading_scores = pd.Series(pca.components_[0], index=selected_data)
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
top_features = sorted_loading_scores.index[:5]
data_subset_reduced = df[top_features]
scaled_data_reduced = StandardScaler().fit_transform(data_subset_reduced)

pca_reduced = PCA()
pca_reduced.fit(scaled_data_reduced)

explained_variance_reduced = pca_reduced.explained_variance_ratio_

# Scree plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.xlabel('Principal Component Number')
plt.ylabel('Eigenvalues')
plt.title('Scree Plot 2')
plt.savefig('ScreePlot2.jpg')
plt.show()

# Determine the number of components to retain
cumulative_variance = np.cumsum(explained_variance)
num_components = np.where(cumulative_variance > 0.95)[0][0] + 1

# Scree plot for the reduced number of principal components
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_components + 1), explained_variance[:num_components], marker='o', linestyle='--')
plt.xlabel('Principal Component Number')
plt.ylabel('Explained Variance')
plt.title('Scree Plot (Reduced Principal Components)')
plt.savefig('ScreePlot_Reduced.jpg')
plt.show()


# Transform the data to the selected number of components
transformed_data = pca.transform(scaled_data)  # Using transform without slicing

# Getting significant features for each of the top principal components
significant_features = []
for i in range(num_components):
    loading_scores = pd.Series(pca.components_[i], index=selected_data)
    sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
    top_feature = sorted_loading_scores.index[0]
    significant_features.append(top_feature)
significant_features = list(set(significant_features))

print("Significant Features for the Top Principal Components:", significant_features)

df_new = df.copy()
df_new.to_csv('churn_cleaned.csv', index=False)

print(df.shape == df_new.shape)

