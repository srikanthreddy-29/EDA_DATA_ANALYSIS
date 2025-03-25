import streamlit as st 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

#Load Titanic dataset 
@st.cache
def load_data():
    data = pd.read_csv("titanic dataset.csv")
    return data

data = load_data()

#Title and description 
st.title('Exploratory analysis of titanic data set')
st.write('This is an EDA on Titanic dataset.')
st.write('First few rows of the dataset:')
st.dataframe(data.head())

#Data Cleaning Section
st.subheader('Missing values')
missing_data = data.isnull().sum()
st.write(missing_data)

if st.checkbox('Fill missing Age with median'):
    data['Age'].fillna(pd.to_numeric(data['Age']).median(),inplace=True)

if st.checkbox('Fill missing Embraked with mode'):
    data['Embraked'].fillna(data['Embraked'].mode()[0],inplace=True)

if st.checkbox('Drop duplicates'):
    data.drop_duplicates(inplace=True)

st.subheader('Cleaned Dataset')
st.dataframe(data.head())

#EDA Section 
st.subheader('Statistical Summary of the Data')
st.write(data.describe())

#Age Distribution
st.subheader('Age Distribution')
fig, ax = plt.subplots()
sns.histplot(data['Age'],kde=True,ax=ax)
ax.set_title('Age Distribution')
st.pyplot(fig)

#Gender Distribution 
st.subheader('Gender Distribution')
fig, ax = plt.subplots()
sns.countplot(x='Sex',data=data,ax=ax)
ax.set_title('Gender Distribution')
st.pyplot(fig)

#Pclass vs Survived
st.subheader('Pclass vs Survived')
fig, ax = plt.subplots()
sns.countplot(x='Pclass',hue='Survived',data=data,ax=ax)
ax.set_title('Pclass vs Survived')
st.pyplot(fig)

'''
#Correlation Heatmap
st.subheader('Correlation Heatmap')
fig, ax = plt.subplots()
sns.heatmap(data.corr(),annot=True,cmap='coolwarm',ax=ax)
ax.set_title('Correlation Heatmap')
st.pyplot(fig)'
'''

#Feature Engineering Section
st.subheader('Feature Engineering: Family Size')
data['Familysize'] = data['SibSp'] + data['Parch']
fig, ax = plt.subplots()
sns.histplot(data['Familysize'],kde=True, ax=ax)
ax.set_title('Family Size Distribution')
st.pyplot(fig)

#Conclusion Section
st.subheader('Key Insights')
insights = """""
-- Females have a higher survival rate than males.\n 
--Passengers in 1st class had the highesst survival rate.\n
--The majority of passengers are in Pclass 3,\n
--Younger passengers tended to survive more often.\n
"""
st.write(insights)




