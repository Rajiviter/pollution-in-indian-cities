#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the  important libraries and the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('9pollutioncsv')


# In[2]:


#convert the dataset into a pandas dataframe
df=pd.DataFrame(data)


# In[3]:


#top 5 entries of the dataframe
df.head()


# In[4]:


#lower 5 entries of dataframe
df.tail()


# In[5]:


#describe the dataframe
df.describe()


# In[6]:


# Calculate the number of unique cities, stations, and states
num_cities = df['city'].nunique()
num_stations = df['station'].nunique()
num_states = df['state'].nunique()

# Print the results
print(f'Number of unique cities: {num_cities}')
print(f'Number of unique stations: {num_stations}')
print(f'Number of unique states: {num_states}')


# In[7]:


# Calculate the number of unique cities, stations, and states
num_cities = df['city'].nunique()
num_stations = df['station'].nunique()
num_states = df['state'].nunique()

# Data for plotting
categories = ['Cities', 'Stations', 'States']
values = [num_cities, num_stations, num_states]

# Plotting
plt.figure(figsize=(8, 6))
plt.bar(categories, values, color=['blue', 'green', 'red'])
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Number of Unique Cities, Stations, and States')
plt.show()


# In[13]:


# Group by 'pollutant_id' and calculate the average pollution levels for each pollutant
df_grouped = df.groupby('pollutant_id')['pollutant_avg'].mean().reset_index()

# Sort by pollutant average in descending order
df_grouped = df_grouped.sort_values(by='pollutant_avg', ascending=False)

# Plot a bar chart
plt.figure(figsize=(10, 6))
plt.bar(df_grouped['pollutant_id'], df_grouped['pollutant_avg'], color=['blue', 'green', 'red'])

# Add titles and labels
plt.title('Comparison of Average Pollution Levels by Pollutant', fontsize=16)
plt.xlabel('Pollutant ID', fontsize=12)
plt.ylabel('Average Pollution Level', fontsize=12)
plt.grid(axis='y')

# Show the plot
plt.tight_layout()
plt.show()


# In[8]:


import seaborn as sns

# Distribution of average pollution levels
plt.figure(figsize=(10, 6))
sns.histplot(df['pollutant_avg'], bins=30, kde=True, color='blue')
plt.title('Distribution of Average Pollution Levels')
plt.xlabel('Average Pollution Level')
plt.ylabel('Frequency')
plt.show()


# In[9]:


# Calculate average pollution levels by pollutant
pollutant_avg = df.groupby('pollutant_id')['pollutant_avg'].mean()

# Plot pollution levels by pollutant
plt.figure(figsize=(10, 6))
pollutant_avg.plot(kind='bar', color='green')
plt.title('Average Pollution Levels by Pollutant')
plt.xlabel('Pollutant')
plt.ylabel('Average Pollution Level')
plt.xticks(rotation=45)
plt.show()


# In[10]:


# Pivot table to have pollutants as columns and average pollution levels as values
pollutant_pivot = df.pivot_table(index='station', columns='pollutant_id', values='pollutant_avg', aggfunc='mean')

# Calculate correlation matrix
correlation_matrix = pollutant_pivot.corr()

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Different Pollutants')
plt.show()


# In[11]:


# Calculate average pollution levels by city
city_avg_pollution = df.groupby('city')['pollutant_avg'].mean()

# Get top 100 cities with highest average pollution
top_100_cities = city_avg_pollution.sort_values(ascending=False).head(100).index

# Filter the dataframe for the top 100 polluted cities
top_100_df = df[df['city'].isin(top_100_cities)]

# Count the number of top 100 cities per state
state_counts = top_100_df['state'].value_counts()

# Get the top 5 states with the highest number of top 100 cities
top_5_states = state_counts.head(5)

# Plot the share of states among the top 5 states
plt.figure(figsize=(10, 7))
top_5_states.plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightpink'])
plt.title('Share of States Among the Top 100 Most Polluted Cities (Top 5 States)')
plt.ylabel('')  # Hide the y-label as it's not needed for a pie chart
plt.show()


# In[16]:


# Calculate average pollution levels by city
city_avg_pollution = df.groupby('city')['pollutant_avg'].mean()

# Get top 100 cities with highest average pollution
top_100_cities = city_avg_pollution.sort_values(ascending=False).head(100).index

# Filter the dataframe for the top 100 polluted cities
top_100_df = df[df['city'].isin(top_100_cities)]

# Count the number of top 100 cities per state
state_counts = top_100_df['state'].value_counts()

# Get the lowest 5 states with the fewest number of top 100 cities
lowest_5_states = state_counts.tail(5)

# Plot the share of states among the lowest 5 states
plt.figure(figsize=(10, 7))
lowest_5_states.plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightpink'])
plt.title('Share of States Among the Lowest 5 States with Fewest Top 100 Polluted Cities')
plt.ylabel('')  # Hide the y-label as it's not needed for a pie chart
plt.show()


# In[12]:


# Calculate average pollution levels by city
city_avg_pollution = df.groupby('city')['pollutant_avg'].mean()

# Get the lowest 100 cities with the lowest average pollution
lowest_100_cities = city_avg_pollution.sort_values(ascending=True).head(100).index

# Filter the dataframe for the lowest 100 polluted cities
lowest_100_df = df[df['city'].isin(lowest_100_cities)]

# Count the number of lowest 100 cities per state
state_counts = lowest_100_df['state'].value_counts()

# Get the top 5 states with the highest number of lowest 100 cities
top_5_states = state_counts.head(5)

# Plot the share of top 5 states
plt.figure(figsize=(10, 7))
top_5_states.plot(kind='pie', autopct='%1.1f%%', colors=plt.get_cmap('tab20').colors)
plt.title('Share of Top 5 States Among the Lowest 100 Most Polluted Cities')
plt.ylabel('')  # Hide the y-label as it's not needed for a pie chart
plt.show()


# In[13]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample DataFrame (replace with your actual data)
# df = pd.read_csv('your_data.csv') 

# Calculate average pollution level per city
city_avg_pollution = df.groupby('city')['pollutant_avg'].mean()

# Define a threshold for highly polluted cities (e.g., top 10% most polluted cities)
threshold = city_avg_pollution.quantile(0.90)
highly_polluted_cities = city_avg_pollution[city_avg_pollution > threshold].index

# Filter original DataFrame for these highly polluted cities
df_highly_polluted = df[df['city'].isin(highly_polluted_cities)]

# Count highly polluted cities by state
state_polluted_cities_count = df_highly_polluted.groupby('state')['city'].nunique()

# Identify the top 10 states with the most highly polluted cities
top_10_states = state_polluted_cities_count.nlargest(10)

# Identify the lowest 10 cities based on their average pollution levels
lowest_10_cities = city_avg_pollution.nsmallest(10)

# Create a DataFrame for heatmap
heatmap_data = pd.DataFrame({
    'City': lowest_10_cities.index,
    'Average Pollution Level': lowest_10_cities.values
})

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Heatmap for top 10 states
heatmap_top_states = pd.DataFrame({
    'State': top_10_states.index,
    'Number of Highly Polluted Cities': top_10_states.values
}).set_index('State').T
sns.heatmap(heatmap_top_states, annot=True, cmap='coolwarm', ax=axes[0])
axes[0].set_title('Top 10 States with Most Highly Polluted Cities')

# Heatmap for lowest 10 cities
heatmap_lowest_cities = heatmap_data.set_index('City').T
sns.heatmap(heatmap_lowest_cities, annot=True, cmap='coolwarm', ax=axes[1])
axes[1].set_title('Lowest 10 Cities with Lowest Pollution Levels')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()


# In[14]:


# Group by city and calculate the average pollution levels for each city
city_avg_pollution = df.groupby('city')['pollutant_avg'].mean().sort_values()

# Get the top 10 and lowest 10 cities
top_10_cities = city_avg_pollution.tail(10)

lowest_10_cities = city_avg_pollution.head(10)

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot for top 10 cities
axes[0].barh(top_10_cities.index, top_10_cities.values, color='green')
axes[0].set_title('Top 10 Cities with Highest Pollution Levels')
axes[0].set_xlabel('Average Pollution Level')
axes[0].set_ylabel('City')

# Plot for lowest 10 cities
axes[1].barh(lowest_10_cities.index, lowest_10_cities.values, color='red')
axes[1].set_title('Top 10 Cities with Lowest Pollution Levels')
axes[1].set_xlabel('Average Pollution Level')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()




# In[14]:


# Calculate average pollution levels by city
city_avg_pollution = df.groupby('city')['pollutant_avg'].mean()

# Get the top 100 cities with the highest average pollution
top_100_cities = city_avg_pollution.sort_values(ascending=False).head(100).index

# Filter the dataframe for the top 100 polluted cities
top_100_df = df[df['city'].isin(top_100_cities)]

# Count the frequency of each pollutant in the top 100 cities
pollutant_counts = top_100_df['pollutant_id'].value_counts()

# Plot the results as a pie chart
plt.figure(figsize=(10, 7))
pollutant_counts.plot(kind='pie', autopct='%1.1f%%', colors=plt.get_cmap('tab20').colors)
plt.title('Distribution of Pollutants in the Top 100 Most Polluted Cities')
plt.ylabel('')  # Hide the y-label as it's not needed for a pie chart
plt.show()


# In[36]:


# Calculate average pollution level per city
city_avg_pollution = df.groupby('city')['pollutant_avg'].mean()

# Identify top 10 and lowest 10 cities based on average pollution levels
top_10_cities = city_avg_pollution.nlargest(10).index
lowest_10_cities = city_avg_pollution.nsmallest(10).index

# Filter the original DataFrame for these cities
df_top_10 = df[df['city'].isin(top_10_cities)]
df_lowest_10 = df[df['city'].isin(lowest_10_cities)]

# Find the most concerning pollutant for each city in the top 10 and lowest 10 cities
top_10_pollutants = df_top_10.groupby('city').apply(lambda x: x.loc[x['pollutant_avg'].idxmax(), 'pollutant_id'])
lowest_10_pollutants = df_lowest_10.groupby('city').apply(lambda x: x.loc[x['pollutant_avg'].idxmax(), 'pollutant_id'])

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot pie chart for top 10 cities
top_10_pollutants.value_counts().plot(kind='pie', ax=axes[0], autopct='%1.1f%%', colors=['blue', 'green', 'red'])
axes[0].set_title('Pollutant Distribution in Top 10 Polluted Cities')
axes[0].set_ylabel('')  # Remove ylabel for pie chart

# Plot pie chart for lowest 10 cities
lowest_10_pollutants.value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%', colors=['orange', 'purple', 'cyan'])
axes[1].set_title('Pollutant Distribution in Lowest 10 Polluted Cities')
axes[1].set_ylabel('')  # Remove ylabel for pie chart

# Adjust layout and show the plot
plt.tight_layout()
plt.show()



# In[15]:


# Calculate the average pollution per station
station_pollution = df.groupby('station')['pollutant_avg'].mean()

# Get the top 10 polluted stations
top_10_stations = station_pollution.sort_values(ascending=False).head(10)

# Plot the results
plt.figure(figsize=(12, 8))
top_10_stations.plot(kind='bar', color='red')
plt.title('Top 10 Most Polluted Stations in India')
plt.xlabel('Station')
plt.ylabel('Average Pollution Level')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[16]:


# Calculate the correlation between latitude and pollution levels
correlation = df['latitude'].corr(df['pollutant_avg'])
print(f'Correlation between latitude and pollution levels: {correlation:.2f}')

# Plot the relationship
plt.figure(figsize=(10, 6))
sns.scatterplot(x='latitude', y='pollutant_avg', data=df, alpha=0.6)
plt.title('Relationship Between Latitude and Pollution Levels')
plt.xlabel('Latitude')
plt.ylabel('Average Pollution Level')
plt.show()


# In[17]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data


# Calculate the correlation between longitude and pollution levels
correlation = df['longitude'].corr(df['pollutant_avg'])
print(f'Correlation between longitude and pollution levels: {correlation:.2f}')

# Plot the relationship
plt.figure(figsize=(10, 6))
sns.scatterplot(x='longitude', y='pollutant_avg', data=df, alpha=0.6)
plt.title('Relationship Between Longitude and Pollution Levels')
plt.xlabel('Longitude')
plt.ylabel('Average Pollution Level')
plt.show()


# In[ ]:




