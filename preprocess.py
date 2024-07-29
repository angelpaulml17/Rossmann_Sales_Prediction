#!/usr/bin/env python
# coding: utf-8

# In[56]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[57]:


store = pd.read_csv('./store (2).csv')
train = pd.read_csv('./train(2) (2).csv')
test = pd.read_csv('./test(1) (2).csv')

store


# In[58]:


train


# In[59]:


test


# In[60]:


store.info()


# In[61]:


train.info()


# In[62]:


test.info()


# In[63]:


store.isna().sum()


# In[64]:


train.isna().sum()


# In[65]:


test.isna().sum()


# In[66]:


merged_df_train = pd.merge(store, train, how='left', on='Store')
merged_df_train


# In[67]:


merged_df_test = pd.merge(store, test, how='right', on='Store')
merged_df_test


# In[68]:


merged_df_test['Date'].unique()


# In[69]:


merged_df_train.isna().sum()


# In[70]:


from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply LabelEncoder to convert alphanumeric data to numerical data
merged_df_train['StoreType'] = label_encoder.fit_transform(merged_df_train['StoreType'])
merged_df_train['Assortment'] = label_encoder.fit_transform(merged_df_train['Assortment'])

merged_df_test['StoreType'] = label_encoder.fit_transform(merged_df_test['StoreType'])
merged_df_test['Assortment'] = label_encoder.fit_transform(merged_df_test['Assortment'])


# In[71]:


merged_df_train


# In[72]:


merged_df_train.describe()


# In[73]:


# Identify rows where Promo2 was filled with 0
fill_zero = merged_df_train['CompetitionDistance'] == 0

# Fill corresponding rows in Promo2SinceWeek and Promo2SinceYear with 0
merged_df_train.loc[fill_zero, 'CompetitionOpenSinceMonth'] = 0
merged_df_train.loc[fill_zero, 'CompetitionOpenSinceYear'] = 0


fill_zero = merged_df_test['CompetitionDistance'] == 0
merged_df_test
# Fill corresponding rows in Promo2SinceWeek and Promo2SinceYear with 0
merged_df_test.loc[fill_zero, 'CompetitionOpenSinceMonth'] = 0
merged_df_test.loc[fill_zero, 'CompetitionOpenSinceYear'] = 0

#Check for null values
merged_df_train.isnull().sum()


# In[74]:


import pandas as pd

# Assuming merged_df_train is your DataFrame

# Group by similarity criteria
grouped_data = merged_df_train.groupby(['StoreType', 'Assortment'])

# Impute missing values for CompetitionOpenSinceYear with the median year of the group
merged_df_train['CompetitionOpenSinceYear'] = merged_df_train['CompetitionOpenSinceYear'].fillna(
    grouped_data['CompetitionOpenSinceYear'].transform('median')
)

# Impute missing values for CompetitionOpenSinceMonth with the median month of the group
merged_df_train['CompetitionOpenSinceMonth'] = merged_df_train['CompetitionOpenSinceMonth'].fillna(
    grouped_data['CompetitionOpenSinceMonth'].transform('median')
)




# Group by similarity criteria
grouped_data = merged_df_test.groupby(['StoreType', 'Assortment'])

# Impute missing values for CompetitionOpenSinceYear with the median year of the group
merged_df_test['CompetitionOpenSinceYear'] = merged_df_test['CompetitionOpenSinceYear'].fillna(
    grouped_data['CompetitionOpenSinceYear'].transform('median')
)

# Impute missing values for CompetitionOpenSinceMonth with the median month of the group
merged_df_test['CompetitionOpenSinceMonth'] = merged_df_test['CompetitionOpenSinceMonth'].fillna(
    grouped_data['CompetitionOpenSinceMonth'].transform('median')
)


# In[75]:


merged_df_train['CompetitionOpenSinceMonth'].isnull()==True


# In[76]:


merged_df_train.isnull().sum()


# In[77]:


# Check if there are still missing values
if merged_df_train['CompetitionOpenSinceYear'].isna().any() or merged_df_train['CompetitionOpenSinceMonth'].isna().any():
    # Second level of imputation
    # Group by StoreType only for a broader grouping
    grouped_data_storetype = merged_df_train.groupby('StoreType')

    # Impute with median based on StoreType
    merged_df_train['CompetitionOpenSinceYear'] = merged_df_train['CompetitionOpenSinceYear'].fillna(
        grouped_data_storetype['CompetitionOpenSinceYear'].transform('median')
    )
    merged_df_train['CompetitionOpenSinceMonth'] = merged_df_train['CompetitionOpenSinceMonth'].fillna(
        grouped_data_storetype['CompetitionOpenSinceMonth'].transform('median')
    )

# If there are still missing values, you can opt for an overall median
overall_median_year = merged_df_train['CompetitionOpenSinceYear'].median()
overall_median_month = merged_df_train['CompetitionOpenSinceMonth'].median()

merged_df_train['CompetitionOpenSinceYear'].fillna(overall_median_year, inplace=True)
merged_df_train['CompetitionOpenSinceMonth'].fillna(overall_median_month, inplace=True)



# Check if there are still missing values
if merged_df_test['CompetitionOpenSinceYear'].isna().any() or merged_df_test['CompetitionOpenSinceMonth'].isna().any():
    # Second level of imputation
    # Group by StoreType only for a broader grouping
    grouped_data_storetype = merged_df_test.groupby('StoreType')

    # Impute with median based on StoreType
    merged_df_test['CompetitionOpenSinceYear'] = merged_df_test['CompetitionOpenSinceYear'].fillna(
        grouped_data_storetype['CompetitionOpenSinceYear'].transform('median')
    )
    merged_df_test['CompetitionOpenSinceMonth'] = merged_df_test['CompetitionOpenSinceMonth'].fillna(
        grouped_data_storetype['CompetitionOpenSinceMonth'].transform('median')
    )

# If there are still missing values, you can opt for an overall median
overall_median_year = merged_df_test['CompetitionOpenSinceYear'].median()
overall_median_month = merged_df_test['CompetitionOpenSinceMonth'].median()

merged_df_test['CompetitionOpenSinceYear'].fillna(overall_median_year, inplace=True)
merged_df_test['CompetitionOpenSinceMonth'].fillna(overall_median_month, inplace=True)


# In[78]:


maxval=4* max(merged_df_train['CompetitionDistance'])
merged_df_train['CompetitionDistance'].fillna(maxval, inplace=True)


maxval=4* max(merged_df_test['CompetitionDistance'])
merged_df_test['CompetitionDistance'].fillna(maxval, inplace=True)


#Check for null values
merged_df_train.isnull().sum()


# In[79]:


merged_df_train.describe()


# In[80]:


# Identify rows where Promo2 was filled with 0
fill_zero = merged_df_train['Promo2'] == 0

# Fill corresponding rows in Promo2SinceWeek and Promo2SinceYear with 0
merged_df_train.loc[fill_zero, 'Promo2SinceWeek'] = 0
merged_df_train.loc[fill_zero, 'Promo2SinceYear'] = 0


fill_zero = merged_df_test['Promo2'] == 0
merged_df_test
# Fill corresponding rows in Promo2SinceWeek and Promo2SinceYear with 0
merged_df_test.loc[fill_zero, 'Promo2SinceWeek'] = 0
merged_df_test.loc[fill_zero, 'Promo2SinceYear'] = 0

#Check for null values
merged_df_train.isnull().sum()


# In[81]:


# Assuming merged_df_train is your DataFrame and it contains the 'PromoInterval' column

# Define the mapping for PromoInterval
promo_interval_mapping = {
    "Jan,Apr,Jul,Oct": 1,
    "Feb,May,Aug,Nov": 2,
    "Mar,Jun,Sept,Dec": 3
}

# Apply the mapping to the 'PromoInterval' column
merged_df_train['PromoIntervalEncoded'] = merged_df_train['PromoInterval'].map(promo_interval_mapping)
merged_df_train

merged_df_test['PromoIntervalEncoded'] = merged_df_test['PromoInterval'].map(promo_interval_mapping)
merged_df_test



# In[82]:


merged_df_train['PromoIntervalEncoded'].fillna(0, inplace=True)
merged_df_train

merged_df_test['PromoIntervalEncoded'].fillna(0, inplace=True)
merged_df_test


# In[83]:


# Assuming df is your pandas DataFrame
duplicates = merged_df_train.duplicated()

# Count the number of duplicate rows
num_duplicates = duplicates.sum()

# Print the number of duplicates
print(f'There are {num_duplicates} duplicate rows in the DataFrame.')

# If you want to see the duplicate rows
if num_duplicates > 0:
    print(df[duplicates])


# In[84]:


merged_df_train['StateHoliday'] = merged_df_train['StateHoliday'].astype(str)
merged_df_test['StateHoliday'] = merged_df_test['StateHoliday'].astype(str)


# In[85]:


holiday_mapping = {
    "a": 1,  # Public holiday
    "b": 2,  # Easter holiday
    "c": 3,  # Christmas
    "0": 0   # None
}

# Apply the mapping to the 'SchoolHoliday' column
merged_df_train['StateHoliday'] = merged_df_train['StateHoliday'].map(holiday_mapping)
merged_df_train

merged_df_test['StateHoliday'] = merged_df_test['StateHoliday'].map(holiday_mapping)
merged_df_test


# In[86]:


merged_df_train.isnull().sum()


# In[87]:


merged_df_train=merged_df_train[~((merged_df_train['Open'] == 0) & (merged_df_train['Promo'] == 1) & (merged_df_train['PromoIntervalEncoded']==0))]


# In[88]:


merged_df_train.isnull().sum()


# In[89]:


merged_df_train[merged_df_train['StateHoliday'].isnull()]


# In[90]:


merged_df_train=merged_df_train[~((merged_df_train['Open'] == 1) & (merged_df_train['Sales'] == 0))]


# In[91]:


merged_df_train


# In[92]:


merged_df_train['SalesperCustomer']=merged_df_train['Sales']/merged_df_train['Customers']


# In[93]:


merged_df_train['Date'] = pd.to_datetime(merged_df_train['Date'], format='%d/%m/%Y')
# Extract date, month, and year
merged_df_train['Day'] = merged_df_train['Date'].dt.day
merged_df_train['Month'] = merged_df_train['Date'].dt.month
merged_df_train['Year'] = merged_df_train['Date'].dt.year


merged_df_test['Date'] = pd.to_datetime(merged_df_test['Date'], format='%d/%m/%Y')

# Extract date, month, and year
merged_df_test['Day'] = merged_df_test['Date'].dt.day
merged_df_test['Month'] = merged_df_test['Date'].dt.month
merged_df_test['Year'] = merged_df_test['Date'].dt.year


# In[94]:


merged_df_test['Date'].unique()


# In[97]:


# Group by year and promotion interval, calculate average sales
sales_by_year_promo = merged_df_train.groupby(['Year', 'PromoInterval'])['Sales'].mean().unstack()
print(sales_by_year_promo)
num_months = 12
years = [2013, 2014, 2015]
promotion_intervals = ['Feb,May,Aug,Nov', 'Jan,Apr,Jul,Oct', 'Mar,Jun,Sept,Dec']

# Plotting
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 18))

for i, year in enumerate(years):
    ax = axes[i]
    sales_by_year_promo.loc[year].plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title(f'Average Sales by Promotion Interval for {year}')
    ax.set_xlabel('Promotion Interval')
    ax.set_ylabel('Average Sales')
    ax.set_xticklabels(promotion_intervals)
    ax.grid(axis='y', linestyle='--', alpha=0.7)


plt.show()


# In[132]:


# Group by 'StoreType' and calculate total sales and count
store_sales = merged_df_train.groupby('StoreType')['Sales'].sum()
store_counts = merged_df_train['StoreType'].value_counts()

# Create a figure and a set of subplots
fig, ax1 = plt.subplots()

# Plot sales data
ax1.bar(store_sales.index, store_sales.values, color='b', alpha=0.6, label='Total Sales')


# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_xlabel('Store Type')
ax1.set_ylabel('Total Sales', color='b')
ax1.tick_params('y', colors='b')

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()

# Plot store count data
ax2.plot(store_counts.index, store_counts.values, color='r', marker='o', label='Store Count')

# Again, make the y-axis label, ticks and tick labels match the line color.
ax2.set_ylabel('Store Count', color='r')
ax2.tick_params('y', colors='r')

# Set the title of the axes
ax1.set_title('Sales and Store Count by Store Type')

# Show the plot
plt.show()


# In[133]:


# Group by 'StoreType' and calculate total sales and count
store_sales = merged_df_train.groupby('Assortment')['Sales'].sum()
store_counts = merged_df_train['Assortment'].value_counts()

# Create a figure and a set of subplots
fig, ax1 = plt.subplots()

# Plot sales data
ax1.bar(store_sales.index, store_sales.values, color='b', alpha=0.6, label='Total Sales')

# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_xlabel('Assortment')
ax1.set_ylabel('Total Sales', color='b')
ax1.tick_params('y', colors='b')

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()

# Plot store count data
ax2.plot(store_counts.index, store_counts.values, color='r', marker='o', label='Store Count')

# Again, make the y-axis label, ticks and tick labels match the line color.
ax2.set_ylabel('Store Count', color='r')
ax2.tick_params('y', colors='r')

# Set the title of the axes
ax1.set_title('Sales and Store Count by Assortment')

# Show the plot
plt.show()


# In[98]:


import matplotlib.pyplot as plt
import pandas as pd

sales_by_year_promo = merged_df_train.groupby(['Year', 'PromoInterval'])['Sales'].mean().unstack()
print(sales_by_year_promo)
num_months = 12
years = [2013, 2014, 2015]
# Plotting the line chart with points
plt.figure(figsize=(10, 6))

# Plotting each promotion interval as a separate line with markers for each year
for promotion_interval in promotion_intervals:
    plt.plot(years, sales_by_year_promo[promotion_interval], '-o', label=promotion_interval)

# Adding title and labels
plt.title('Average Sales by Promotion Interval and Year')
plt.xlabel('Year')
plt.ylabel('Average Sales')
plt.xticks(years)  # Ensure only the years in the dataset are used as x-ticks
plt.grid(True)
plt.legend(title='PromoInterval')

# Display the plot
plt.show()


# In[96]:


# Group sales by year and month
sales_by_month = merged_df_train.groupby(['Year', 'Month']).agg({'Sales': 'sum'}).reset_index()

# Plotting
fig, axes = plt.subplots(nrows=len(years), ncols=1, figsize=(10, 4*len(years)))

for i, year in enumerate(years):
    ax = axes[i]
    sales_year = sales_by_month[sales_by_month['Year'] == year]
    ax.bar(sales_year['Month'], sales_year['Sales'], color='skyblue')
    ax.set_title(f'Sales by Month for {year}')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Sales')
    ax.set_xticks(range(1, 13))
    ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


# In[48]:


# Group by promotion interval and calculate average sales
sales_by_promotion_interval = merged_df_train.groupby('PromoInterval')['Sales'].mean()

# Define the order of promotion intervals
promotion_interval_order = ['Jan,Apr,Jul,Oct', 'Feb,May,Aug,Nov', 'Mar,Jun,Sept,Dec']

# Plotting
plt.figure(figsize=(10, 6))
sales_by_promotion_interval[promotion_interval_order].plot(kind='bar', color='skyblue')
plt.title('Average Sales by Promotion Interval')
plt.xlabel('Promotion Interval')
plt.ylabel('Average Sales')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[49]:


merged_df_train


# In[50]:


merged_df_train[(merged_df_train['Store']==2) & (merged_df_train['PromoIntervalEncoded']==1) &(merged_df_train['Year']==2013)]


# In[99]:


def comp_months(df):
    
    df['CompOpenSince'] = 12 * (df.Year - df.CompetitionOpenSinceYear) + (df.Month - df.CompetitionOpenSinceMonth)
    df['CompOpenSince'] = df['CompOpenSince'].map(lambda x: 0 if x < 0 else x).fillna(0) #to account for the negative values we will get for stores built in future
    
comp_months(merged_df_train)


# In[100]:


def comp_months(df):
    mask = (df['CompetitionDistance'] != 0)  # Use '&' instead of 'and'
    df.loc[mask, 'CompOpenSince'] = 12 * (df.loc[mask, 'Year'] - df.loc[mask, 'CompetitionOpenSinceYear']) + \
                                    (df.loc[mask, 'Month'] - df.loc[mask, 'CompetitionOpenSinceMonth'])
    df['CompOpenSince'] = df['CompOpenSince'].map(lambda x: max(0, x)).fillna(0)  # Ensure non-negative values

# Call the function with the DataFrame
comp_months(merged_df_train)
comp_months(merged_df_test)


# In[182]:


merged_df_train


# In[101]:


merged_df_train['WeekofYear'] = merged_df_train['Date'].dt.isocalendar().week
merged_df_test['WeekofYear'] = merged_df_test['Date'].dt.isocalendar().week
def promo_cols(df):
    # Months since Promo2 was open
    df['Promo2Open'] = 12 * (df.Year - df.Promo2SinceYear) +  (df.WeekofYear - df.Promo2SinceWeek)*7/30  #Average of 30 and 31 days
    df['Promo2Open'].fillna(0, inplace =True) 
    df['Promo2Open'] = df['Promo2Open'].map(lambda x: 0 if x < 0 else x).fillna(0) * df['Promo2']
promo_cols(merged_df_train)
promo_cols(merged_df_test)


# In[107]:


merged_df_train


# In[102]:


# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(merged_df_train['CompOpenSince'], merged_df_train['Sales'], color='skyblue', alpha=0.5)
plt.title('Effect of Time Since Competition Open on Sales')
plt.xlabel('Months Since Competition Open')
plt.ylabel('Sales')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[55]:


# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(merged_df_train['Promo2Open'], merged_df_train['Sales'], color='skyblue', alpha=0.5)
plt.title('Effect of Time Since promotion Open on Sales')
plt.xlabel('Months Since promotion Open')
plt.ylabel('Sales')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[56]:


unique_values = merged_df_train['CompOpenSince'].unique()

# Print unique values
print(unique_values.astype(int))


# In[57]:


merged_df_train[(merged_df_train['CompetitionDistance']== 303440.0) & (merged_df_train['StoreType']==0) & (merged_df_train['Assortment']==2)]


# In[58]:


merged_df_train[(merged_df_train['CompetitionDistance']== 0)]


# In[288]:


# # Get total sales, customers and open days per store
# store_data_sales = merged_df_train.groupby([merged_df_train['Store']])['Sales'].sum()
# store_data_customers = merged_df_train.groupby([merged_df_train['Store']])['Customers'].sum()
# store_data_open = merged_df_train.groupby([merged_df_train['Store']])['Open'].count()
     

# # Calculate sales per day, customers per day and sales per customers per day
# store_data_sales_per_day = store_data_sales / store_data_open
# store_data_customers_per_day = store_data_customers / store_data_open
# store_data_sales_per_customer_per_day = store_data_sales_per_day / store_data_customers_per_day
     

# #Saving the above values in a dictionary so that they can be mapped to the dataframe.
# sales_per_day_dict = dict(store_data_sales_per_day)
# customers_per_day_dict = dict(store_data_customers_per_day)
# sales_per_customers_per_day_dict = dict(store_data_sales_per_customer_per_day)

# merged_df_train['SalesPerDay'] = merged_df_train['Store'].map(sales_per_day_dict)
# merged_df_train['Customers_per_day'] = merged_df_train['Store'].map(customers_per_day_dict)
# merged_df_train['Sales_Per_Customers_Per_Day'] = merged_df_train['Store'].map(sales_per_customers_per_day_dict)


# In[289]:


merged_df_train


# ## Feature adding

# In[103]:


states = pd.read_csv('./store_states.csv')
merged_df_train= pd.merge(merged_df_train, states, how='left', on='Store')
merged_df_test= pd.merge(merged_df_test, states, how='left', on='Store')


# In[104]:


merged_df_train['State'].unique()


# In[105]:


merged_df_train


# In[106]:


merged_df_train['PromoAndHolidayInteraction'] = merged_df_train['Promo'].astype(str) + '_' + merged_df_train['StateHoliday'].astype(str) + '_' + merged_df_train['SchoolHoliday'].astype(str)

merged_df_test['PromoAndHolidayInteraction'] = merged_df_test['Promo'].astype(str) + '_' + merged_df_test['StateHoliday'].astype(str) + '_' + merged_df_test['SchoolHoliday'].astype(str)


# In[107]:


sales_by_interaction = merged_df_train.groupby('PromoAndHolidayInteraction')['Sales'].sum()

# Plotting
plt.figure(figsize=(10, 6))
sales_by_interaction.plot(kind='bar', color='skyblue')
plt.title('Total Sales by Promo and Holiday Interaction')
plt.xlabel('Promo and Holiday Interaction')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[64]:


merged_df_train


# In[108]:


# Assuming df is your DataFrame
# Calculate the correlation matrix
corr_matrix = merged_df_train.corr()

# Plot the heatmap
plt.figure(figsize=(30, 20))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .5})
plt.title('Correlation Matrix')
plt.show()


# In[109]:


#Columns to be dropped
columns_to_drop = ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'WeekofYear']

# Drop the columns
merged_df_train_mod = merged_df_train.drop(columns=columns_to_drop)
merged_df_train_mod

# Drop the columns
merged_df_test_mod = merged_df_test.drop(columns=columns_to_drop)
merged_df_test_mod


# In[110]:


merged_df_train.isna().sum()


# In[193]:


merged_df_train[merged_df_train['Sales'].isna()]


# In[71]:


# First, ensure the 'Date' column is in datetime format
merged_df_train['Date'] = pd.to_datetime(merged_df_train['Date'])

# Identify public holidays directly by 'StateHoliday' column
# Assuming 'a', 'b', 'c' represent different public holidays and '0' represents non-holidays
public_holidays = merged_df_train_mod[merged_df_train_mod['StateHoliday'].isin([1,2,3])]['Date'].unique()
public_holidays = pd.to_datetime(public_holidays)
# Assuming you have a Series or list of public_holidays as datetime objects
public_holidays_df = pd.DataFrame({'PublicHolidayDate': pd.to_datetime(public_holidays).unique()})
public_holidays_df.sort_values('PublicHolidayDate', inplace=True)
# Ensure both DataFrames are sorted by the date columns
merged_df_train_mod.sort_values('Date', inplace=True)
public_holidays_df.sort_values('PublicHolidayDate', inplace=True)

# Find the nearest previous public holiday for each date in merged_df_train
nearest_previous_holiday = pd.merge_asof(merged_df_train_mod, public_holidays_df,
                                         left_on='Date', right_on='PublicHolidayDate',
                                         direction='backward')

# Calculate the days from the nearest previous public holiday
nearest_previous_holiday['DaysFromPreviousPublicHoliday'] = (nearest_previous_holiday['Date'] - nearest_previous_holiday['PublicHolidayDate']).dt.days



# Assuming 'DaysFromPreviousPublicHoliday' and 'DaysUntilNextPublicHoliday' do not exist in merged_df_train
merged_df_train_mod['DaysFromPreviousPublicHoliday'] = nearest_previous_holiday['DaysFromPreviousPublicHoliday']



# In[72]:


# First, ensure the 'Date' column is in datetime format
merged_df_test['Date'] = pd.to_datetime(merged_df_test['Date'])

# Identify public holidays directly by 'StateHoliday' column
# Assuming 'a', 'b', 'c' represent different public holidays and '0' represents non-holidays
public_holidays = merged_df_test_mod[merged_df_test_mod['StateHoliday'].isin([1,2,3])]['Date'].unique()
public_holidays = pd.to_datetime(public_holidays)
# Assuming you have a Series or list of public_holidays as datetime objects
public_holidays_df = pd.DataFrame({'PublicHolidayDate': pd.to_datetime(public_holidays).unique()})
public_holidays_df.sort_values('PublicHolidayDate', inplace=True)
# Ensure both DataFrames are sorted by the date columns
merged_df_test_mod.sort_values('Date', inplace=True)
public_holidays_df.sort_values('PublicHolidayDate', inplace=True)

# Find the nearest previous public holiday for each date in merged_df_test
nearest_previous_holiday = pd.merge_asof(merged_df_test_mod, public_holidays_df,
                                         left_on='Date', right_on='PublicHolidayDate',
                                         direction='backward')

# Calculate the days from the nearest previous public holiday
nearest_previous_holiday['DaysFromPreviousPublicHoliday'] = (nearest_previous_holiday['Date'] - nearest_previous_holiday['PublicHolidayDate']).dt.days



# Assuming 'DaysFromPreviousPublicHoliday' and 'DaysUntilNextPublicHoliday' do not exist in merged_df_test
merged_df_test_mod['DaysFromPreviousPublicHoliday'] = nearest_previous_holiday['DaysFromPreviousPublicHoliday']



# In[73]:


# Ensure merged_df_train_mod is sorted in ascending order by 'Date'
merged_df_train_mod = merged_df_train_mod.sort_values('Date', ascending=True)

# Ensure public_holidays_df_reversed is sorted in ascending order by 'PublicHolidayDate'
public_holidays_df_reversed = public_holidays_df.sort_values('PublicHolidayDate', ascending=True)

# Now perform the merge_asof
nearest_next_holiday_reversed = pd.merge_asof(merged_df_train_mod, public_holidays_df,
                                              left_on='Date', right_on='PublicHolidayDate',
                                              direction='forward')  # Consider using 'forward' if looking for the next holiday

# Calculate the days until the next public holiday
nearest_next_holiday_reversed['DaysUntilNextPublicHoliday'] = (nearest_next_holiday_reversed['PublicHolidayDate'] - nearest_next_holiday_reversed['Date']).dt.days

# Assuming nearest_next_holiday should be nearest_next_holiday_reversed in the assignment statement
merged_df_train_mod['DaysUntilNextPublicHoliday'] = nearest_next_holiday_reversed['DaysUntilNextPublicHoliday']


# In[74]:


# Ensure merged_df_test_mod is sorted in ascending order by 'Date'
merged_df_test_mod = merged_df_test_mod.sort_values('Date', ascending=True)

# Ensure public_holidays_df_reversed is sorted in ascending order by 'PublicHolidayDate'
public_holidays_df_reversed = public_holidays_df.sort_values('PublicHolidayDate', ascending=True)

# Now perform the merge_asof
nearest_next_holiday_reversed = pd.merge_asof(merged_df_test_mod, public_holidays_df,
                                              left_on='Date', right_on='PublicHolidayDate',
                                              direction='forward')  # Consider using 'forward' if looking for the next holiday

# Calculate the days until the next public holiday
nearest_next_holiday_reversed['DaysUntilNextPublicHoliday'] = (nearest_next_holiday_reversed['PublicHolidayDate'] - nearest_next_holiday_reversed['Date']).dt.days

# Assuming nearest_next_holiday should be nearest_next_holiday_reversed in the assignment statement
merged_df_test_mod['DaysUntilNextPublicHoliday'] = nearest_next_holiday_reversed['DaysUntilNextPublicHoliday']


# In[75]:


def find_days_until_next_holiday(row_date, holidays_series):
    # Filter for holidays that are after the current row's date
    future_holidays = holidays_series[holidays_series > row_date]
    if not future_holidays.empty:
        # Find the closest future holiday
        next_holiday = future_holidays.iloc[0]
        return (next_holiday - row_date).days
    else:
        # Return NaN or a placeholder if there is no next holiday
        return np.nan

# Ensure public_holidays_df and merged_df_train_mod are sorted in ascending order
public_holidays_df = public_holidays_df.sort_values('PublicHolidayDate')
merged_df_train_mod = merged_df_train_mod.sort_values('Date')

# Apply the custom function to each row in your main DataFrame
merged_df_train_mod['DaysUntilNextPublicHoliday'] = merged_df_train_mod['Date'].apply(
    lambda x: find_days_until_next_holiday(x, public_holidays_df['PublicHolidayDate']))


# In[76]:


def find_days_until_next_holiday(row_date, holidays_series):
    # Filter for holidays that are after the current row's date
    future_holidays = holidays_series[holidays_series > row_date]
    if not future_holidays.empty:
        # Find the closest future holiday
        next_holiday = future_holidays.iloc[0]
        return (next_holiday - row_date).days
    else:
        # Return NaN or a placeholder if there is no next holiday
        return np.nan

# Ensure public_holidays_df and merged_df_test_mod are sorted in ascending order
public_holidays_df = public_holidays_df.sort_values('PublicHolidayDate')
merged_df_test_mod = merged_df_test_mod.sort_values('Date')

# Apply the custom function to each row in your main DataFrame
merged_df_test_mod['DaysUntilNextPublicHoliday'] = merged_df_test_mod['Date'].apply(
    lambda x: find_days_until_next_holiday(x, public_holidays_df['PublicHolidayDate']))


# In[77]:


merged_df_train_mod_rev = merged_df_train_mod.sort_values('Date', ascending=False)
merged_df_train_mod_rev


# In[79]:


merged_df_train_mod.isna().sum()


# In[80]:


merged_df_train_mod.sort_values('Store', ascending=True)


# In[111]:


# Assuming df is your DataFrame
# Calculate the correlation matrix
corr_matrix = merged_df_train_mod.corr()

# Plot the heatmap
plt.figure(figsize=(30, 20))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .5})
plt.title('Correlation Matrix')
plt.show()


# In[112]:


merged_df_trained=merged_df_train_mod.copy()
merged_df_tested=merged_df_test_mod.copy()


# In[113]:


corr_matrix = merged_df_trained.corr()

# Plot the heatmap
plt.figure(figsize=(30, 20))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .5})
plt.title('Correlation Matrix')
plt.show()


# In[87]:


merged_df_trained['SalesperCustomer'].fillna(0, inplace=True)
merged_df_trained['DaysUntilNextPublicHoliday'].fillna(0, inplace=True)

#merged_df_tested['SalesperCustomer'].fillna(0, inplace=True)
merged_df_tested['DaysUntilNextPublicHoliday'].fillna(0, inplace=True)


# In[114]:


label_encoder = LabelEncoder()

# Apply LabelEncoder to convert alphanumeric data to numerical data
merged_df_trained['State'] = label_encoder.fit_transform(merged_df_trained['State'])

merged_df_tested['State'] = label_encoder.fit_transform(merged_df_tested['State'])


# In[115]:


merged_df_trained.columns


# In[90]:


from sklearn.linear_model import LinearRegression

# Assuming df is your DataFrame and it's already cleaned and contains only numeric variables

def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    vif_data["VIF"] = 0.0

    for i in range(len(df.columns)):
        X = df.drop(df.columns[i], axis=1)
        y = df[df.columns[i]]
        
        # Fit linear regression model
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate R-squared
        r_squared = model.score(X, y)
        
        # Calculate VIF
        if r_squared < 1:  # Prevent division by zero
            vif = 1 / (1 - r_squared)
        else:
            vif = np.inf
        vif_data.loc[i, "VIF"] = vif
    
    return vif_data

# Calculate VIF for the DataFrame
vif_df = calculate_vif(numeric_df)  # Ensure df_numeric is your numeric DataFrame
print(vif_df)


# In[342]:


merged_df_trained


# ## Feature selection

# In[91]:


merged_df_trained['Year'] = merged_df_trained['Date'].dt.year
merged_df_trained['Day'] = merged_df_trained['Date'].dt.day

merged_df_tested['Year'] = merged_df_tested['Date'].dt.year
merged_df_tested['Day'] = merged_df_tested['Date'].dt.day


# In[198]:


merged_df_trained.columns


# In[141]:


merged_df_trained_1= merged_df_trained.drop(columns=['Date'])


# In[128]:


# merged_df_trained_1=merged_df_trained_1.drop(columns=['Date'])
merged_df_trained_1.columns


# In[139]:


merged_df_tested_1= merged_df_tested.drop(columns=['Date'])


# In[147]:


merged_df_trained_1= merged_df_trained.drop(columns=['Date'])
merged_df_trained_1= merged_df_trained_1.drop(columns=['SalesperCustomer'])
merged_df_trained_1


# In[148]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Assuming df is your DataFrame and 'target' is the name of your target column
X = merged_df_trained_1.drop(['Customers', 'Sales'], axis=1)
y = merged_df_trained_1['Customers']

# Encode categorical variables if necessary
# X = pd.get_dummies(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[120]:


merged_df_trained_1.columns


# In[352]:


from sklearn.tree import DecisionTreeClassifier  # or DecisionTreeRegressor for regression problems

# Initialize and fit the Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Get feature importances
feature_importances_dt = pd.DataFrame(dt.feature_importances_, index = X_train.columns, columns=['Importance']).sort_values('Importance', ascending=False)

print("Feature importances from Decision Tree:")
print(feature_importances_dt)


# In[ ]:





# ## MOdel

# In[356]:


from statsmodels.tsa.statespace.sarimax import SARIMAX

# Assuming df is indexed by Date and includes 'Sales' as the target variable
# and 'Customers' and 'Promo' as explanatory variables.

# Define the target variable and exogenous predictors
y = merged_df_trained_1['Sales']
X = merged_df_trained_1.drop(columns=['Sales', 'Customers']) # Including 'Customers' and 'Promo' as exogenous variables

# Fit a SARIMAX model
model = SARIMAX(y, exog=X, order=(1, 1, 1), seasonal_order=(1, 1, 1, 30))
results = model.fit()


# In[357]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Assume X, y have been preprocessed and are ready for modeling
# X is the feature set, y is the target variable (e.g., Sales)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model on the training data
rf_regressor.fit(X_train, y_train)

# Predict on the test data
rf_predictions = rf_regressor.predict(X_test)

# Evaluate the model (e.g., using RMSE)
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y_test, rf_predictions, squared=False)
print(f"RMSE: {rmse}")


# In[381]:


from sklearn.metrics import mean_squared_error, r2_score
# Calculate R² (coefficient of determination)
y_pred = xgb_regressor.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f'R²: {r2}')


# In[205]:


merged_df_trained_1


# In[160]:


import xgboost as xgb
label_encoder = LabelEncoder()

merged_df_trained_1['DayOfWeek']= label_encoder.fit_transform(merged_df_trained_1['DayOfWeek'])
merged_df_trained_1['PromoAndHolidayInteraction'] = label_encoder.fit_transform(merged_df_trained_1['PromoAndHolidayInteraction'])
merged_df_trained_2=merged_df_trained_1[merged_df_trained_1['Open']==1]
y = merged_df_trained_2['Sales']
X = merged_df_trained_2.drop(columns=['Sales', 'Customers']) # Including 'Customers' and 'Promo' as exogenous variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the XGBoost Regressor
xgb_regressor_sales = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, enable_categorical=True)

# Fit the model on the training data
xgb_regressor_sales.fit(X_train, y_train)

# Predict on the test data
xgb_predictions = xgb_regressor_sales.predict(X_test)

rmspe = np.sqrt(np.mean(((y_test - xgb_predictions) / y_test) ** 2)) * 100
print(f"RMSPE: {rmspe}%")


# In[131]:


merged_df_trained_1.columns


# In[161]:


y = merged_df_trained_2['Customers']
X = merged_df_trained_2.drop(columns=['Sales', 'Customers']) # Including 'Customers' and 'Promo' as exogenous variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the XGBoost Regressor
xgb_regressor_customers = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, enable_categorical=True)

# Fit the model on the training data
xgb_regressor_customers.fit(X_train, y_train)

# Predict on the test data
xgb_predictions = xgb_regressor_customers.predict(X_test)

rmspe = np.sqrt(np.mean(((y_test - xgb_predictions) / y_test) ** 2)) * 100
print(f"RMSPE: {rmspe}%")


# In[ ]:


xgb_predictions = xgb_regressor.predict(X_test)


# In[138]:


merged_df_trained_1.columns


# In[151]:


merged_df_tested_1


# In[162]:


merged_df_tested_1['DayOfWeek']= label_encoder.fit_transform(merged_df_tested_1['DayOfWeek'])
merged_df_tested_1['PromoAndHolidayInteraction'] = label_encoder.fit_transform(merged_df_tested_1['PromoAndHolidayInteraction'])
merged_df_tested_2 = merged_df_tested_1[merged_df_tested_1['Open'] == 1]
X_testset = merged_df_tested_2.drop(['Customers', 'Sales'], axis=1)
xgb_predictions_sales = xgb_regressor_customers.predict(X_testset)

# Step 2: Initialize 'predicted_customers' in the original DataFrame with NaNs
merged_df_tested_1['predicted_customers'] = np.nan

# Step 3: Update 'predicted_customers' with the predictions where 'Open' is 1
merged_df_tested_1.loc[merged_df_tested_2.index, 'predicted_customers'] = xgb_predictions_sales

# Step 4: Set 'Sales' and 'Customers' to 0 where 'Open' is 0
# This assumes you want to keep the original 'Sales' and 'Customers' columns unchanged for 'Open' == 1 rows
# If 'Sales' and 'Customers' are not needed for 'Open' == 0, you can assign 0 directly
merged_df_tested_1.loc[merged_df_tested_1['Open'] == 0, ['Sales', 'Customers']] = 0


# In[166]:


merged_df_tested_1['date'] = pd.to_datetime(merged_df_tested_1[['Year', 'Month', 'Day']])

# Then, group by this new 'date' column to get daily sales
daily_sales = merged_df_tested_1.groupby('date')['predicted_sales'].sum().reset_index()

# Plotting daily predicted sales
plt.figure(figsize=(15, 7))
plt.plot(daily_sales['date'], daily_sales['predicted_sales'], marker='o', linestyle='-')
plt.title('Daily Predicted Sales')
plt.xlabel('Date')
plt.ylabel('Predicted Sales')
plt.xticks(rotation=45)  # Rotate dates for better readability
plt.grid(True)
plt.show()


# In[239]:


merged_df_tested_1['date'].unique()


# In[164]:


merged_df_tested_2 = merged_df_tested_1[merged_df_tested_1['Open'] == 1]
X_testset = merged_df_tested_2.drop(['Customers', 'Sales', 'predicted_customers'], axis=1)
xgb_predictions_sales = xgb_regressor_sales.predict(X_testset)

# Step 2: Initialize 'predicted_customers' in the original DataFrame with NaNs
merged_df_tested_1['predicted_sales'] = np.nan

# Step 3: Update 'predicted_customers' with the predictions where 'Open' is 1
merged_df_tested_1.loc[merged_df_tested_2.index, 'predicted_sales'] = xgb_predictions_sales


# In[159]:


merged_df_tested_1= merged_df_tested.drop(columns=['Date'])
merged_df_tested_1


# In[165]:


merged_df_tested_1['date'] = pd.to_datetime(merged_df_tested_1[['Year', 'Month', 'Day']])

# Then, group by this new 'date' column to get daily sales
daily_sales = merged_df_tested_1.groupby('date')['predicted_customers'].sum().reset_index()

# Plotting daily predicted sales
plt.figure(figsize=(15, 7))
plt.plot(daily_sales['date'], daily_sales['predicted_customers'], marker='o', linestyle='-')
plt.title('Daily Predicted Customers')
plt.xlabel('Date')
plt.ylabel('Predicted Customers')
plt.xticks(rotation=45)  # Rotate dates for better readability
plt.grid(True)
plt.show()


# In[223]:


xgb_predictions


# In[226]:


merged_df_tested_1[merged_df_tested_1['Open']==0]


# In[ ]:


xgb_predictions = xgb_regressor.predict(X_test)


# In[170]:


# Group by date to get daily sales and customers
merged_df_tested_1['Date'] = pd.to_datetime(merged_df_tested_1['date'], format='%Y-%m-%d')
daily_data = merged_df_tested_1.groupby('Date').agg({
    'predicted_sales': 'sum',
    'predicted_customers': 'sum'
}).reset_index()

# Plotting daily predicted sales and customers
plt.figure(figsize=(15, 7))

# First axis for predicted sales
ax1 = plt.gca()  # Get current axis
ax2 = ax1.twinx()  # Create another axis that shares the same x-axis

# Plot predicted sales
ax1.plot(daily_data['Date'], daily_data['predicted_sales'], color='blue', marker='o', linestyle='-', label='Predicted Sales')
ax1.set_xlabel('Date')
ax1.set_ylabel('Predicted Sales', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xticklabels(daily_data['Date'], rotation=45)  # Rotate dates for better readability

# Plot predicted customers on the same plot but with a different y-axis
ax2.plot(daily_data['Date'], daily_data['predicted_customers'], color='green', marker='x', linestyle='-', label='Predicted Customers')
ax2.set_ylabel('Predicted Customers', color='green')  
ax2.tick_params(axis='y', labelcolor='green')

# Additional plot settings
plt.title('Daily Predicted Sales and Customers')
plt.grid(True)

# Legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='best')

plt.show()


# In[174]:


# Group by 'date' to get daily aggregates
daily_data = merged_df_tested_1.groupby('date').agg({
    'predicted_sales': 'sum',
    'predicted_customers': 'sum'
}).reset_index()

# Ensure 'date' is in datetime format for plotting (it should already be, but just to be sure)
daily_data['date'] = pd.to_datetime(daily_data['date'])

# Plotting
plt.figure(figsize=(15, 7))
ax1 = plt.gca()  # Primary axis for predicted sales
ax2 = ax1.twinx()  # Secondary axis for predicted customers

# Predicted sales
ax1.plot(daily_data['date'], daily_data['predicted_sales'], color='blue', marker='o', linestyle='-', label='Predicted Sales')
ax1.set_xlabel('Date')
ax1.set_ylabel('Predicted Sales', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Predicted customers
ax2.plot(daily_data['date'], daily_data['predicted_customers'], color='green', marker='x', linestyle='-', label='Predicted Customers')
ax2.set_ylabel('Predicted Customers', color='green')
ax2.tick_params(axis='y', labelcolor='green')

# Format the x-axis to display dates correctly
plt.gcf().autofmt_xdate()  # Auto format for date rotation and alignment
date_format = mpl_dates.DateFormatter('%Y-%m-%d')
ax1.xaxis.set_major_formatter(date_format)

plt.title('Daily Predicted Sales and Customers')
plt.grid(True)

# Combining legends from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='best')

plt.show()


# In[177]:


import pandas as pd

# Assuming 'merged_df_tested_1' is your DataFrame and it has a 'date' column
# Step 1: Convert 'date' column to datetime format if it's not already # Adjust format as necessary
merged_df_trained_1['date'] = pd.to_datetime(merged_df_trained_1[['Year', 'Month', 'Day']])
# Step 2: Create a complete date range from the min to the max date in your dataset
min_date = merged_df_trained_1['date'].min()
max_date = merged_df_trained_1['date'].max()
all_dates = pd.date_range(start=min_date, end=max_date, freq='D')

# Step 3: Identify missing dates
# Convert the 'date' column to a DatetimeIndex
existing_dates = pd.DatetimeIndex(merged_df_trained_1['date'])

# Find the difference between the complete date range and the existing dates
missing_dates = all_dates.difference(existing_dates)

# missing_dates now contains all the dates that were not present in your 'date' column
print(missing_dates)


# In[178]:


merged_df_trained_1


# In[ ]:





# In[ ]:





# In[ ]:





# In[113]:


epsilon = 1e-10  # A small number to avoid division by zero
rmspe = np.sqrt(np.mean(((y_test - xgb_predictions) / (y_test + epsilon))**2)) * 100
print(f"RMSPE: {rmspe}%")


# In[214]:


from sklearn.metrics import mean_squared_error
def best_params(**params):
    model_X= xgb.XGBRegressor(**params, random_state=33,n_jobs=-1).fit(X_train, y_train)
    predict_train= model_X.predict(X_train)
    predict_val= model_X.predict(X_test)
    rmse_train=mean_squared_error(predict_train, y_train, squared= False)
    rmse_val=mean_squared_error(predict_val, y_test, squared=False)
    
    print(f"With Hyperparameter tuning - \nTraining Error: {rmse_train} \nValidation Error: {rmse_val}")

#Fitting with best parameters (I manually tried different combinations of parameters)
best_params(n_estimators=300, max_depth=12, learning_rate=0.3, subsample=0.75, colsample_bytree=0.77)


# In[215]:


importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_regressor.feature_importances_
}).sort_values('importance', ascending=False)
plt.figure(figsize=(10,6))
plt.title('Feature Importance')
sns.barplot(data=importance_df.head(10), x='importance', y='feature');


# In[105]:


X.columns


# In[216]:


# Import necessary libraries
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# It's often a good idea to standardize the data before applying PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Instantiate PCA object for 2 principal components
pca = PCA(n_components=9)

# Fit PCA on the data
pca.fit(X_scaled)

# Transform the data to its principal components
X_pca = pca.transform(X_scaled)

# Print the transformed data
print("Original shape:", X_scaled.shape)
print("Transformed shape:", X_pca.shape)
print("Principal Components:\n", X_pca)


# In[217]:


y = merged_df_trained_1['Sales']
 # Including 'Customers' and 'Promo' as exogenous variables
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
# Initialize the XGBoost Regressor
xgb_regressor = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, enable_categorical=True)

# Fit the model on the training data
xgb_regressor.fit(X_train, y_train)

# Predict on the test data
xgb_predictions = xgb_regressor.predict(X_test)

rmspe = np.sqrt(np.mean(((y_test - xgb_predictions) / y_test) ** 2)) * 100
print(f"RMSPE: {rmspe}%")


# In[129]:


from sklearn.metrics import mean_squared_error
def best_params(**params):
    model_X= xgb.XGBRegressor(**params, random_state=33,n_jobs=-1).fit(X_train, y_train)
    predict_train= model_X.predict(X_train)
    predict_val= model_X.predict(X_test)
    rmse_train=mean_squared_error(predict_train, y_train, squared= False)
    rmse_val=mean_squared_error(predict_val, y_test, squared=False)
    
    print(f"With Hyperparameter tuning - \nTraining Error: {rmse_train} \nValidation Error: {rmse_val}")

#Fitting with best parameters (I manually tried different combinations of parameters)
best_params(n_estimators=300, max_depth=12, learning_rate=0.3, subsample=0.75, colsample_bytree=0.77)


# In[ ]:


import pandas as pd

# Assuming 'merged_df_tested_1' is your DataFrame and it has a 'date' column
# Step 1: Convert 'date' column to datetime format if it's not already
merged_df_trained_1['date'] = pd.to_datetime(merged_df_trained_1['date'], format='%Y-%m-%d')  # Adjust format as necessary

# Step 2: Create a complete date range from the min to the max date in your dataset
min_date = merged_df_trained_1['date'].min()
max_date = merged_df_trained_1['date'].max()
all_dates = pd.date_range(start=min_date, end=max_date, freq='D')

# Step 3: Identify missing dates
# Convert the 'date' column to a DatetimeIndex
existing_dates = pd.DatetimeIndex(merged_df_trained_1['date'])

# Find the difference between the complete date range and the existing dates
missing_dates = all_dates.difference(existing_dates)

# missing_dates now contains all the dates that were not present in your 'date' column
print(missing_dates)


# In[176]:


merged_df_trained_1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[364]:


get_ipython().system('pip install xgboost')


# In[369]:


merged_df_trained_1.info()


# In[ ]:





# In[ ]:





# In[310]:


# Calculate average sales for school holiday vs non-school holiday
average_sales = merged_df_train_mod.groupby('SchoolHoliday')['Sales'].mean().reset_index()

# Plotting
plt.figure(figsize=(8, 6))
sns.barplot(data=average_sales, x='SchoolHoliday', y='Sales')
plt.title('Average Sales: School Holiday vs Non-School Holiday')
plt.xlabel('School Holiday (0 = No, 1 = Yes)')
plt.ylabel('Average Sales')
plt.xticks([0, 1], ['Non-Holiday', 'Holiday'])
plt.show()


# In[140]:


merged_df_train_mod_filtered = merged_df_train_mod[merged_df_train_mod['StateHoliday'] != 0]
#Step 5: Count occurrences of each holiday type for each state
holiday_counts = merged_df_train_mod_filtered.groupby(['State', 'StateHoliday']).size().unstack(fill_value=0)

# Step 6: Plot the stacked bar graph
holiday_counts.plot(kind='bar', stacked=True, colormap='viridis')

# Step 7: Add labels and title
plt.xlabel('States')
plt.ylabel('Number of Holidays')
plt.title('Number of Holidays in Different States')
plt.legend(title='Holiday Types')

# Step 8: Show the graph
plt.show()


# In[311]:


# Calculate average sales for each state holiday category
average_sales_by_state_holiday = merged_df_train_mod.groupby('StateHoliday')['Sales'].mean().reset_index()

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(data=average_sales_by_state_holiday, x='StateHoliday', y='Sales', palette="Set2")
plt.title('Average Sales by State Holiday Type')
plt.xlabel('State Holiday Type')
plt.ylabel('Average Sales')
plt.xticks(rotation=45)  # Rotate labels if they overlap or are too long
plt.show()


# In[309]:


merged_df_train_mod[(merged_df_train_mod['Promo2']==0) & merged_df_train_mod['Promo2Interval']!=1]


# In[313]:


plt.figure(figsize=(10, 6))
sns.barplot(data=merged_df_train_mod, x='DayOfWeek', y='Sales', hue='Promo', palette="coolwarm",
            ci=None, order=[1, 2, 3, 4, 5, 6, 7])  # ci=None to remove confidence interval bars for clarity
plt.title('Effect of Promotion on Sales by Day of the Week')
plt.xlabel('Day of the Week (1=Monday, 7=Sunday)')
plt.ylabel('Average Sales')
plt.legend(title='Promo', loc='upper right', labels=['No Promo', 'Promo'])

plt.show()


# In[315]:


plt.figure(figsize=(10, 6))
sns.barplot(data=merged_df_train_mod, x='StoreType', y='Sales', hue='Promo', palette="viridis",
            ci=None, order=[0,1,2,3])  # ci=None to remove confidence interval bars for clarity
plt.title('Effect of Promotion on Sales by Store Type')
plt.xlabel('Store Type')
plt.ylabel('Average Sales')
plt.legend(title='Promo', loc='upper right', labels=['No Promo', 'With Promo'])

plt.show()


# In[134]:


# Scatter plot showing individual stores
plt.figure(figsize=(10, 6))
plt.scatter(merged_df_train_mod['CompetitionDistance'], merged_df_train_mod['Sales'], alpha=0.5)
plt.title('Sales vs Competition Distance')
plt.xlabel('Competition Distance (meters)')
plt.ylabel('Sales')
plt.grid(True)
plt.show()


# In[316]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df is your DataFrame

# Create a FacetGrid
g = sns.FacetGrid(merged_df_train_mod, col="StoreType", hue="Promo", col_wrap=2, height=4, aspect=1.5, palette="viridis", sharey=False)

# Map a bar plot for sales vs. day of the week onto each facet
g.map(sns.barplot, "DayOfWeek", "Sales", order=range(1, 8), ci=None)

# Add a legend and titles
g.add_legend(title="Promo")
g.set_titles("{col_name} StoreType")
g.set_axis_labels("Day of the Week", "Average Sales")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Effect of Promotion on Sales by Day of the Week and Store Type')

plt.show()


# In[322]:


merged_df_train_mod['DayOfWeek'] = merged_df_train_mod['DayOfWeek'].astype(str)

# Setting the figure size
plt.figure(figsize=(14, 8))

# Create pointplot
sns.pointplot(data=merged_df_train_mod, x='DayOfWeek', y='Sales', hue='StoreType', palette='deep', 
              dodge=True, markers=['o', 's'], linestyles=['-', '--'],
              ci=None)

# Enhancing the plot
plt.title('Impact of Promotion on Sales Across Days of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Average Sales')
plt.legend(title='Promotion Status', labels=['No Promo', 'With Promo'])

plt.tight_layout()
plt.show()


# In[318]:


# Create a FacetGrid
g = sns.FacetGrid(merged_df_train_mod, col="Assortment", hue="Promo", col_wrap=2, height=4, aspect=1.5, palette="viridis", sharey=False)

# Map a bar plot for sales vs. day of the week onto each facet
g.map(sns.barplot, "DayOfWeek", "Sales", order=range(1, 8), ci=None)

# Add a legend and titles
g.add_legend(title="Promo")
g.set_titles("{col_name} Assortment")
g.set_axis_labels("Day of the Week", "Average Sales")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Effect of Promotion on Sales by Day of the Week and Assortment')

plt.show()


# ### Data Visualization

# In[42]:


# Create a line chart with StoreType and Promo combination
sns.catplot(x="DayOfWeek", y="Sales", hue="StoreType", col="Promo", data=merged_df_train, kind="point", height=6, aspect=1.5)

# Set labels and title
plt.xlabel("Day of Week")
plt.ylabel("Average Sales")
plt.suptitle("Average Sales by Day of Week for Each Store Type and Promo Combination", y=1.05)

# Show the plot
plt.show()


# In[50]:


import seaborn as sns
# Visualize the distribution of the Outcome variable
sns.countplot(x='Sales', data=merged_df_train)
plt.title('Distribution of Sales')
plt.show()


# In[51]:


# Create a boxplot
plt.figure(figsize=(15, 10))
ax = sns.boxplot(data=merged_df_train.drop(['Sales', 'Customers'], axis=1))
ax.set_yscale('log')  # Set y-axis to logarithmic scale as the dataset has highly varied data
# Show the plot
plt.show()


# In[52]:


# Pairplot to visualize relationships between variables
sns.pairplot(merged_df_train, hue='Sales', diag_kind='kde')
plt.suptitle('Pairplot of Variables', y=1.02)
plt.show()


# In[53]:


plt.figure(figsize=(4,4))
sns.countplot(merged_df_train, x= 'Promo2')
plt.title("Stores running Promo 2")
plt.xticks([0,1], ['No', 'Yes']);


# In[55]:


plt.figure(figsize=(4,5))
sns.kdeplot(merged_df_train['CompetitionDistance']/1000 )
plt.title("Competitor Distance Distribution in km")


# In[56]:


plt.figure(figsize=(4,4))
sns.countplot(merged_df_train, x= 'Promo')
plt.title("Stores running a Promo")
plt.xticks([0,1], ['No', 'Yes']);


# In[59]:


plt.figure(figsize=(4,4))
sns.barplot(merged_df_train, x='Promo', y='Sales')


# In[179]:


plt.figure(figsize=(4,4))
sns.barplot(merged_df_train, x='CompetitionDistance', y='Sales')


# In[181]:


# Binning competition distances into categories
bins = [0, 1000, 2000, 3000, 4000, 5000, 10000, 20000, float('inf')]
labels = ['0-1000', '1001-2000', '2001-3000', '3001-4000', '4001-5000', '5001-10000', '10001-20000', '20001+']
merged_df_train['CompetitionDistanceGroup'] = pd.cut(merged_df_train['CompetitionDistance'], bins=bins, labels=labels)

# Calculate average sales for each competition distance category
avg_sales_by_distance = merged_df_train.groupby('CompetitionDistanceGroup')['Sales'].mean()

# Create a bar plot for sales based on competition distance
plt.figure(figsize=(10, 6))
avg_sales_by_distance.plot(kind='bar', color='skyblue')
plt.xlabel('Competition Distance Group')
plt.ylabel('Average Sales')
plt.title('Average Sales Based on Competition Distance')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# In[182]:


sales_by_store_type = merged_df_train.groupby('StoreType')['Sales'].sum()

# Create a bar plot for sales based on store type
plt.figure(figsize=(8, 6))
sales_by_store_type.plot(kind='bar', color='skyblue')
plt.xlabel('Store Type')
plt.ylabel('Total Sales')
plt.title('Total Sales by Store Type')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# In[184]:


# Calculate total sales for each assortment type
sales_by_assortment = merged_df_train.groupby('Assortment')['Sales'].sum()

# Plot sales based on assortment
plt.figure(figsize=(8, 6))
sales_by_assortment.plot(kind='bar', color='skyblue')
plt.xlabel('Assortment')
plt.ylabel('Total Sales')
plt.title('Total Sales Based on Assortment')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# In[60]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming df is your DataFrame and it has columns 'Date' and 'Sales'
merged_df_train['Date'] = pd.to_datetime(merged_df_train['Date'])
merged_df_train.sort_values('Date', inplace=True)

plt.figure(figsize=(15,6))
plt.plot(merged_df_train['Date'], merged_df_train['Sales'])
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()


# In[62]:


plt.figure(figsize=(10,6))
plt.hist(merged_df_train['Sales'], bins=50)
plt.title('Sales Distribution')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()


# In[63]:


plt.figure(figsize=(10,6))
plt.scatter(merged_df_train['Customers'], merged_df_train['Sales'])
plt.title('Sales vs Customers')
plt.xlabel('Customers')
plt.ylabel('Sales')
plt.show()


# In[64]:


promo_df = merged_df_train.groupby('Promo')['Sales'].mean().reset_index()

plt.bar(promo_df['Promo'], promo_df['Sales'])
plt.title('Average Sales: Promotion Days vs Non-Promotion Days')
plt.xlabel('Promotion (0 = No, 1 = Yes)')
plt.ylabel('Average Sales')
plt.show()


# In[65]:


store_sales = merged_df_train.groupby('Store')['Sales'].sum().sort_values(ascending=False)

plt.figure(figsize=(15,6))
store_sales.head(10).plot(kind='bar')  # Top 10 stores
plt.title('Top 10 Stores by Sales')
plt.xlabel('Store ID')
plt.ylabel('Total Sales')
plt.show()


# In[66]:


holiday_sales = merged_df_train.groupby('StateHoliday')['Sales'].mean().reset_index()

plt.bar(holiday_sales['StateHoliday'], holiday_sales['Sales'])
plt.title('Sales on Holidays vs Regular Days')
plt.xlabel('State Holiday')
plt.ylabel('Average Sales')
plt.show()


# In[72]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Assuming df is your DataFrame with 'Date' and 'Sales'
train['Date'] = pd.to_datetime(train['Date'])
train.set_index('Date', inplace=True)

# Resample to weekly sales
weekly_sales = train['Sales'].resample('W').sum()

plt.figure(figsize=(15,6))
plt.stem(weekly_sales.index, weekly_sales.values, use_line_collection=True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%W'))
plt.title('Weekly Sales Trends')
plt.xlabel('Week')
plt.ylabel('Total Sales')
plt.show()


# In[68]:


# Resample to monthly sales
monthly_sales = merged_df_train['Sales'].resample('M').sum()

plt.figure(figsize=(15,6))
plt.plot(monthly_sales.index, monthly_sales.values)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.title('Monthly Sales Trends')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.show()


# In[75]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Assuming df is your DataFrame with 'Date' and 'Sales'
train['Date'] = pd.to_datetime(train['Date'])
train.set_index('Date', inplace=True)

# Resample to get weekly sales
weekly_sales = train['Sales'].resample('W').sum()

plt.figure(figsize=(15,6))
plt.stem(weekly_sales.index, weekly_sales.values, use_line_collection=True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%W'))
plt.title('Weekly Sales Trends')
plt.xlabel('Week')
plt.ylabel('Total Sales')
plt.show()


# In[78]:


# Resample to get weekly sales
weekly_sales = train['Sales'].resample('W').sum()

plt.figure(figsize=(15,6))
plt.stem(weekly_sales.index, weekly_sales.values, use_line_collection=True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%W'))
plt.title('Weekly Sales Trends')
plt.xlabel('Week')
plt.ylabel('Total Sales')
plt.show()


# In[77]:


# Resample to get weekly sales
weekly_sales = train['Sales'].resample('D').sum()

plt.figure(figsize=(15,6))
plt.stem(weekly_sales.index, weekly_sales.values, use_line_collection=True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%M'))
plt.title('Weekly Sales Trends')
plt.xlabel('Week')
plt.ylabel('Total Sales')
plt.show()


# In[90]:


# Assuming df is your DataFrame with 'Date' and 'Sales'
train['Date'] = pd.to_datetime(train['Date'])


plt.figure(figsize=(15,6))
plt.stem(train['Date'], train['Sales'], use_line_collection=True)
plt.title('Daily Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.tight_layout()  # Adjust the layout
plt.show()


# In[88]:


train


# In[93]:


import plotly.express as px

# Assuming df is your DataFrame with 'Date' and 'Sales'
# # Ensure 'Date' is a datetime type
# train['Date'] = train.to_datetime(train['Date'])

fig = px.line(train, x='Date', y='Sales', title='Daily Sales with Range Slider')

# Add range slider
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)

fig.show()


# In[118]:


# Filter data for the year 2015
year_to_plot = 2015
store_to_plot = 105

filtered_df = train[(train['Date'].dt.year == year_to_plot) & (train['Store'] == store_to_plot)]
print(filtered_df)
plt.figure(figsize=(15, 6))
plt.stem(filtered_df['Date'], filtered_df['Sales'], use_line_collection=True)
plt.title(f'Stem Plot for Sales in {year_to_plot}')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[129]:


# Filter data for the year 2015
year_to_plot = 2014
store_to_plot = 78
monthstart=1
monthend=3
filtered_df = train[(train['Date'].dt.year == year_to_plot) & ((train['Date'].dt.month >= monthstart) & (train['Date'].dt.month <= monthend)) & (train['Store'] == store_to_plot)]
print(filtered_df)
plt.figure(figsize=(15, 6))
plt.stem(filtered_df['Date'], filtered_df['Sales'], use_line_collection=True)
plt.title(f'Stem Plot for Sales in {year_to_plot}')
plt.xlabel('Date')
plt.ylabel('Sales')

# Set x-ticks to every date in the filtered DataFrame
plt.xticks(filtered_df['Date'], rotation=90)

# To make the plot more readable, you might consider only showing every nth date
# For example, to show every 7th date (roughly weekly):
#plt.xticks(filtered_df['Date'][::7], rotation=45)

plt.tight_layout()
plt.show()


# In[109]:


rolling_mean = train['Sales'].rolling(window=7).mean()

plt.figure(figsize=(15, 6))
plt.plot(train.index, train['Sales'], label='Actual Sales', alpha=0.7)
plt.plot(rolling_mean.index, rolling_mean, label='7-Day Rolling Mean', color='red')

plt.title('Sales with 7-Day Rolling Mean')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# In[142]:


import matplotlib.pyplot as plt

# Filter data for the year 2015
year_to_plot = 2014
store_to_plot = 74
store_to_compare = 110
monthstart = 1
monthend = 3

# Filter data for the 78th store
filtered_df_78 = train[(train['Date'].dt.year == year_to_plot) & 
                        ((train['Date'].dt.month >= monthstart) & (train['Date'].dt.month <= monthend)) & 
                        (train['Store'] == store_to_plot)]

# Filter data for the 112th store
filtered_df_112 = train[(train['Date'].dt.year == year_to_plot) & 
                         ((train['Date'].dt.month >= monthstart) & (train['Date'].dt.month <= monthend)) & 
                         (train['Store'] == store_to_compare)]

plt.figure(figsize=(15, 6))

# Plot sales data for the 78th store in blue
plt.stem(filtered_df_78['Date'], filtered_df_78['Sales'], linefmt='b-', markerfmt='bo', basefmt=' ')

# Plot sales data for the 110th store in red
plt.stem(filtered_df_112['Date'], filtered_df_112['Sales'], linefmt='r-', markerfmt='ro', basefmt=' ')

plt.title(f'Stem Plot for Sales in {year_to_plot}')
plt.xlabel('Date')
plt.ylabel('Sales')

# Set x-ticks to every date in the filtered DataFrame
plt.xticks(filtered_df_78['Date'], rotation=90)

# To make the plot more readable, you might consider only showing every nth date
# For example, to show every 7th date (roughly weekly):
# plt.xticks(filtered_df_78['Date'][::7], rotation=45)

plt.tight_layout()
plt.show()


# In[ ]:





# In[146]:


#Create subplots for three different graphs
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 18))

# Create a list of months in chronological order
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# Group the data by year and month and calculate total sales per month for each graph
for i, ax in enumerate(axes):
   year_to_plot = 2013 + i  # Start with 2014 and increment for each graph
   monthly_sales = train[train['Date'].dt.year == year_to_plot].groupby(train['Date'].dt.month)['Sales'].sum()
   
   # Plot the trend of sales per month as a scatter plot with lines for each year
   ax.plot(months, monthly_sales, marker='o', linestyle='-', color='b', label=f'Year {year_to_plot}')
   ax.set_xlabel('Month')
   ax.set_ylabel('Total Sales')
   ax.set_title(f'Trend of Sales for Year {year_to_plot}')
   ax.grid(True)
   ax.legend()

# Adjust spacing between subplots
plt.tight_layout()
plt.show()


# In[149]:


from statsmodels.tsa.seasonal import seasonal_decompose
# train['Date'] = pd.to_datetime(train['Date'])
# train.set_index('Date', inplace=True)
# Aggregate sales data by day (modify this based on your data structure)
daily_sales = train['Sales'].resample('D').sum()

# Basic visual inspection
plt.figure(figsize=(15, 5))
plt.plot(daily_sales)
plt.title('Daily Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

# Seasonal Decomposition
decomposition = seasonal_decompose(daily_sales, model='additive', period=365)  # adjust period based on expected seasonality
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(daily_sales, label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


# In[186]:


#Make sure to resample the sales data to get daily frequency if it's not already daily

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL

# train['Date'] = pd.to_datetime(train['Date'])
# train.set_index('Date', inplace=True)
train_2014 = train[train.index.year == 2014]
daily_sales_2014 = train_2014['Sales'].resample('D').sum()

# Seasonal Decompose with a period that matches expected seasonality within the year
# For example, if you expect monthly seasonality, you might set the period to 30
decompose_result_2014 = seasonal_decompose(daily_sales_2014, model='additive', period=15)

# STL Decomposition with a similar seasonal period assumption
stl_2014 = STL(daily_sales_2014, seasonal=15, robust=True)
stl_result_2014 = stl_2014.fit()

# Plotting the Seasonal Decompose for 2014
plt.figure(figsize=(14, 10))
decompose_result_2014.plot()
plt.suptitle('Seasonal Decomposition for 2014', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.xticks(rotation=90)
plt.show()


# In[183]:


train


# In[177]:


train_2013 = train[train.index.year == 2013]
# Make sure to resample the sales data to get daily frequency if it's not already daily
daily_sales_2013 = train_2013['Sales'].resample('D').sum()


# Seasonal Decompose with a period that matches expected seasonality within the year
# For example, if you expect monthly seasonality, you might set the period to 30
decompose_result_2013 = seasonal_decompose(daily_sales_2013, model='additive', period=30)

# STL Decomposition with a similar seasonal period assumption
stl_2013 = STL(daily_sales_2013, seasonal=29, robust=True)
stl_result_2013 = stl_2013.fit()

# Plotting the Seasonal Decompose for 2014
plt.figure(figsize=(14, 10))
decompose_result_2013.plot()
plt.suptitle('Seasonal Decomposition for 2013', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.xticks(rotation=90)
plt.show()


# In[198]:


train['Date'] = pd.to_datetime(train['Date'])
train.set_index('Date', inplace=True)

train_2015 = train[train.index.year == 2015]
# Make sure to resample the sales data to get daily frequency if it's not already daily
daily_sales_2015 = train_2015['Sales'].resample('D').sum()


# Seasonal Decompose with a period that matches expected seasonality within the year
# For example, if you expect monthly seasonality, you might set the period to 30
decompose_result_2015 = seasonal_decompose(daily_sales_2015, model='additive', period=29)

# STL Decomposition with a similar seasonal period assumption
stl_2015 = STL(daily_sales_2015, seasonal=29, robust=True)
stl_result_2013 = stl_2013.fit()

# Plotting the Seasonal Decompose for 2014
plt.figure(figsize=(14, 10))
decompose_result_2015.plot()
plt.suptitle('Seasonal Decomposition for 2015', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.xticks(rotation=90)
plt.show()


# In[175]:


# Define the start and end dates
start_date = '2013-01-01'
end_date = '2015-07-31'

# Create a date range covering the entire period
full_date_range = pd.date_range(start=start_date, end=end_date)

# Assuming your dataset has a 'Date' column, convert it to datetime if not already done
merged_df_train['Date'] = pd.to_datetime(merged_df_train['Date'])

# Extract unique dates from your dataset
existing_dates = merged_df_train['Date'].dt.date.unique()

# Convert to set for faster comparison
existing_dates_set = set(existing_dates)

# Find missing dates by comparing with the full date range
missing_dates = [date for date in full_date_range if date.date() not in existing_dates_set]

print("Missing Dates:")
print(missing_dates)


# In[187]:


merged_df_train


# In[ ]:




