# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,notebook-scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Initial Data Exploration

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
pd.set_option('display.max_columns', None)

# %% [markdown]
# First, read in the data with the correct data types and concatenate the training and test data.

# %%
titanic_dtypes = {
    'Survived': bool,
    'Pclass': 'category',
    'Name': str,
    'Sex': 'category',
    'Ticket': str,
    'Cabin': str,
    'Embarked': 'category'
}

# %%
train = pd.read_csv('../data/raw/train.csv', index_col='PassengerId', dtype=titanic_dtypes)
test = pd.read_csv('../data/raw/test.csv', index_col='PassengerId', dtype=titanic_dtypes)

# %%
df = pd.concat([train, test], axis=0, keys=['train', 'test'])

# %%
df.dtypes

# %% [markdown]
# `Pclass` and `Embarked` are currently encoded as single characters. To make plots more helpful, we'll replace these codes with informative levels.

# %%
df['Pclass'] = (
    df['Pclass']
        .cat.rename_categories({'1': 'Upper', '2': 'Middle', '3': 'Lower'})
        .cat.as_ordered()
        .cat.reorder_categories(['Lower', 'Middle', 'Upper'])
)

# %%
df['Embarked'] = (
    df['Embarked']
        .cat.rename_categories({'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'})
)

# %% [markdown]
# ## Data at a glance

# %%
df.head()

# %%
for source, group in df.groupby(level=0):
    print(f'{source}:')
    print('======')
    print(group.info())
    print('\n')

# %%
df.groupby(level=0).apply(lambda x: x.describe())

# %%
train = df.loc['train']
test = df.loc['test']

# %% [markdown]
# ## Target variable: `Survived`

# %%
train['Survived'].value_counts()

# %%
sns.countplot(x='Survived', data=df)

# %% [markdown]
# ## `Pclass` - Ticket Class

# %%
train['Pclass'].value_counts().sort_index()

# %%
sns.catplot(x='Pclass', y='Survived', kind='bar', data=train)

# %% [markdown]
# ## `Sex`

# %%
train['Sex'].value_counts()

# %%
sns.catplot(x='Sex', y='Survived', kind='bar', data=train)

# %% [markdown]
# ## `Age`

# %%
sns.catplot(x='Survived', y='Age', kind='violin', bw=0.25, data=train)

# %% [markdown]
# ## `SibSp` - Number of siblings and spouses aboard

# %%
train['SibSp'].value_counts().sort_index()

# %%
sns.catplot(x='SibSp', y='Survived', kind='bar', data=train)

# %% [markdown]
# ## `Parch` - Number of parents and children aboard

# %%
train['Parch'].value_counts().sort_index()

# %%
sns.catplot(x='Parch', y='Survived', kind='bar', data=train)

# %% [markdown]
# ## `Fare`

# %%
train['Fare'].describe()

# %%
sns.catplot(x='Survived', y='Fare', kind='violin', bw=0.25, data=train)

# %% [markdown]
# ## `Embarked` - Port of Embarkation

# %%
train['Embarked'].value_counts()

# %%
sns.catplot(x='Embarked', y='Survived', kind='bar', data=train)

# %% [markdown]
# ## Correlated Features

# %% [markdown]
# ### Class and Fare

# %%
sns.catplot(x='Pclass', y='Fare', hue='Survived', kind='boxen', data=train)

# %%
sns.catplot(x='Embarked', hue='Pclass', kind='count', data=train)

# %%
sns.catplot(x='Pclass', hue='Embarked', kind='count', data=train)

# %%
sns.catplot(x='Embarked', y='Survived', hue='Pclass', kind='bar', data=train)

# %%
pd.crosstab(train['Pclass'], train['Embarked'], margins=True)

# %%
sns.catplot(x='Survived', y='Age', hue='Sex', kind='violin', data=train)

# %%
