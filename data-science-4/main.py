#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (KBinsDiscretizer,
                                   OneHotEncoder,
                                   StandardScaler)

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import (CountVectorizer,
                                             TfidfVectorizer)


# In[ ]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

# from IPython.core.pylabtools import figsize


# figsize(12, 8)

# sns.set()


# In[ ]:


countries = pd.read_csv("countries.csv")


# In[ ]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[ ]:


df = countries


# In[ ]:


df.info()


# In[ ]:


# Remove espaços a mais
df['Country'] = [country.strip() for country in df['Country']]
df['Region'] = [region.strip() for region in df['Region']]


# In[ ]:


df.head()


# In[ ]:


df.replace(',','.', regex=True, inplace=True)


# In[ ]:


df.head(5)


# In[ ]:


float_cols = ['Pop_density', 'Coastline_ratio', 'Net_migration', 'Infant_mortality', 'GDP',
               'Literacy', 'Phones_per_1000', 'Arable', 'Crops', 'Other', 'Climate',
               'Birthrate', 'Deathrate', 'Agriculture', 'Industry', 'Service']


# In[ ]:


# Transforma em float as strings numéricas
for col in float_cols:
    df[col] = [float(value) for value in df[col]]


# In[ ]:


df.info()


# In[ ]:


# Lista de regiões presentes no dataset

regions = df['Region'].unique()
regions.sort()


# In[ ]:


df['Pop_density'].head()


# In[ ]:


# Quantos países estão no 90º percentil?

bins = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
pop_bins = bins.fit_transform(df[['Pop_density']])
(pop_bins >= 9).sum()


# In[ ]:


# Quantos one-hot-encodes são criados se encodadas as features Region e Climate

encoder = OneHotEncoder()
region_climate_encoded = encoder.fit_transform(df[['Region', 'Climate']].fillna(0))
print(region_climate_encoded.shape)
print(encoder.categories_)


# In[ ]:


### Q4 - Pipeline
# Preencher int64 e float64 com as medianas
df.info()


# In[ ]:


df.isna().sum()


# In[ ]:


pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy="median")),
                           ('scaler', StandardScaler())])

cols_num = df.select_dtypes(include=['int64','float64']).columns
cols_object = df.select_dtypes(exclude=['int64','float64']).columns
pipeline.fit(df[cols_num])


# In[ ]:


####
# Outliers

df['Net_migration'].plot(kind='box')


# In[ ]:


quantil1, quantil3 = df['Net_migration'].quantile([.25, .75])
quantil_interval = quantil3 - quantil1
quantil_interval


# In[ ]:


interval = (quantil1 - 1.5 * quantil_interval, quantil3 + 1.5 * quantil_interval)
interval


# In[ ]:


low_outliers = df['Net_migration'][df['Net_migration'] < interval[0]]
high_outliers = df['Net_migration'][df['Net_migration'] > interval[1]]


# In[ ]:


print(f"Outliers [Inf|Sup]\n[{len(low_outliers)}|{len(high_outliers)}]")
print(f"Total de outliers: {len(low_outliers)+len(high_outliers)}")
print(f"Total de observações: {df.shape[0]}")
print(f"Porcentagem de outliers: {(len(low_outliers)+len(high_outliers))/df.shape[0]}")


# ### fetch_20newsgroups

# In[ ]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[ ]:


count_vect = CountVectorizer()
word_counts = count_vect.fit_transform(newsgroup.data)
print(f"Indice da palavra Phone: {count_vect.vocabulary_.get('phone')}")
print(f"Ocorrencias: {word_counts[:-1, count_vect.vocabulary_.get('phone')].toarray().sum()}")


# In[ ]:


tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(newsgroup.data)
print(tfidf_matrix.shape)
print(tfidf_matrix[:, count_vect.vocabulary_.get('phone')].toarray().sum())


# ----------------------------------------------------
# # QUESTÕES

# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[ ]:


def q1():
    regions = df['Region'].unique()
    regions.sort()
    return list(regions)

q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[ ]:


def q2():
    bins = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    pop_bins = bins.fit_transform(df[['Pop_density']])
    return int((pop_bins >= 9).sum())

q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[ ]:


def q3():
    encoder = OneHotEncoder()
    region_climate_encoded = encoder.fit_transform(df[['Region', 'Climate']].fillna(0))
    return int(region_climate_encoded.shape[1])

q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[ ]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[ ]:


test_country_df = pd.DataFrame([test_country], columns=df.columns)

pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy="median")),
                           ('scaler', StandardScaler())])

cols_num = df.select_dtypes(include=['int64','float64']).columns
cols_object = df.select_dtypes(exclude=['int64','float64']).columns
pipeline.fit(df[cols_num])


# In[ ]:


test_country_transform = pipeline.transform(test_country_df[cols_num])
test_country_transform = pd.DataFrame(test_country_transform, columns=cols_num)
test_country_transform


# In[ ]:


float(round(test_country_transform.Arable, 3))


# In[ ]:


def q4():
    test_country_df = pd.DataFrame([test_country], columns=df.columns)

    pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy="median")),
                               ('scaler', StandardScaler())])

    cols_num = df.select_dtypes(include=['int64','float64']).columns
    cols_object = df.select_dtypes(exclude=['int64','float64']).columns
    pipeline.fit(df[cols_num])
    test_country_transform = pipeline.transform(test_country_df[cols_num])
    test_country_transform = pd.DataFrame(test_country_transform, columns=cols_num)
    return float(round(test_country_transform.Arable, 3))

q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[ ]:


def q5():
    quantil1, quantil3 = df['Net_migration'].quantile([.25, .75])
    quantil_interval = quantil3 - quantil1
    interval = (quantil1 - 1.5 * quantil_interval, quantil3 + 1.5 * quantil_interval)
    low_outliers = df['Net_migration'][df['Net_migration'] < interval[0]]
    high_outliers = df['Net_migration'][df['Net_migration'] > interval[1]]
    return (len(low_outliers), len(high_outliers), False)

q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[ ]:


def q6():
    count_vect = CountVectorizer()
    word_counts = count_vect.fit_transform(newsgroup.data)
    return word_counts[:-1, count_vect.vocabulary_.get('phone')].toarray().sum()

q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[ ]:


def q7():
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(newsgroup.data)
    return float(round(tfidf_matrix[:, count_vect.vocabulary_.get('phone')].toarray().sum(), 3))

q7()

