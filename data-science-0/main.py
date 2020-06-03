#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


black_friday.head()


# In[4]:


# Purchase antes de transformação
black_friday['Purchase'].describe()


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[5]:


def q1():
    return black_friday.shape


# In[6]:


q1()


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[7]:


def q2():
    return black_friday.query('Gender == "F" and Age == "26-35"').shape[0]


# In[8]:


q2()


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[9]:


def q3():
    return black_friday['User_ID'].nunique()


# In[10]:


q3()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[11]:


def q4():
    return len(black_friday.dtypes.value_counts())


# In[12]:


q4()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[13]:


def q5():
    return black_friday.isna().T.any().sum()/len(black_friday.index)


# In[14]:


q5()


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[15]:


def q6():
    nan_count = dict(black_friday.count(axis='rows'))
    values = nan_count.values()
    nan_num = max(values) - min(values)
    return nan_num


# In[16]:


q6()


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[17]:


def q7():
    return black_friday['Product_Category_3'].value_counts().index[0]


# In[18]:


q7()


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[19]:


def q8():
    purchase = black_friday['Purchase']
    norm_purchase = (purchase - purchase.min()) / (purchase.max() - purchase.min())
    return float(norm_purchase.mean())


# In[20]:


q8()


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[21]:


def q9():
    purchase = black_friday['Purchase']
    standard_purchase = (purchase - purchase.mean()) / (purchase.std())
    return int(standard_purchase[(standard_purchase >= -1) & (standard_purchase <= 1)].shape[0])


# In[22]:


q9()


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[23]:


def q10():
    nan_mask = black_friday['Product_Category_2'].isna()
    products_subset = black_friday.loc[:,['Product_Category_2','Product_Category_3']]
    return products_subset[nan_mask].count()[0] == products_subset[nan_mask].count()[1]


# In[24]:


q10()

