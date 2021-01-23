#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd

df = pd.read_csv("C:/Users/Abdel/Desktop/Nouveau dossier/data-final.csv", delimiter='\t')


# In[3]:


df


# In[5]:


columns=df.columns


# In[6]:


for column in columns:
    print(column)


# In[7]:


import numpy as np
#Next we will take the first 50 coluns only as the rest is meta data like the width and height of the screen of the device people have used to fill the questionnaire

x = df[df.columns[0:50]]


# In[8]:


x
#The columns represent questions people they have answered to analyse their personality 


# In[9]:


pd.set_option('display.max_columns', None)


# In[10]:


x = x.fillna(0)
x
#Show the whole new data frame 


# In[13]:


from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=10,random_state=0,batch_size=100,max_iter=100).fit(x)


# In[14]:


len(kmeans.cluster_centers_)


# In[16]:


one = kmeans.cluster_centers_[0]


# In[17]:


two = kmeans.cluster_centers_[1]


# In[18]:


three = kmeans.cluster_centers_[2]


# In[19]:


four = kmeans.cluster_centers_[3]


# In[20]:


five = kmeans.cluster_centers_[4]


# In[21]:


six = kmeans.cluster_centers_[5]


# In[22]:


seven = kmeans.cluster_centers_[6]


# In[23]:


eight = kmeans.cluster_centers_[7]


# In[24]:


nine = kmeans.cluster_centers_[8]


# In[25]:


ten = kmeans.cluster_centers_[9]


# In[26]:


one


# In[32]:


one_score = {}

one_score["extroversion_score"]=-one[0]-one[1]-one[2]-one[3]+one[4]+one[5]+one[6]-one[7]+one[8]+one[9]
one_score["neuroticism_score"]=one[0]+one[1]+one[2]+one[3]-one[4]-one[5]-one[6]-one[7]-one[8]+one[9]
one_score["agreeableness_score"]=-one[0]-one[1]+one[2]-one[3]-one[4]-one[5]-one[6]+one[7]+one[8]-one[9]
one_score["conscientiousness_score"]=-one[0]-one[1]+one[2]-one[3]-one[4]+one[5]+one[6]-one[7]+one[8]+one[9]
one_score["openness_score"]=one[0]-one[1]+one[2]-one[3]+one[4]-one[5]+one[6]-one[7]+one[8]-one[9]


# In[34]:


one_score


# In[61]:


all_types={"one":one, "two":two, "three":three, "four":four, 
           "five":five, "six":six, "seven":seven, "eight":eight, "nine":nine,"ten":ten}


all_types_scores={}

all_types_scores


# In[75]:




for name, personality_type in all_types.items():
    personality_trait = {}

    personality_trait['extroversion_score'] =  personality_type[0] - personality_type[1] +personality_type[2] - personality_type[3] + personality_type[4] - personality_type[5] +personality_type[6] - personality_type[7] + personality_type[8] -personality_type[9]
    personality_trait['neuroticism_score'] =  personality_type[0] - personality_type[1] + personality_type[2] -personality_type[3] + personality_type[4] + personality_type[5] + personality_type[6] + personality_type[7] + personality_type[8] + personality_type[9]
    personality_trait['agreeableness_score'] =  -personality_type[0] +personality_type[1] - personality_type[2] + personality_type[3] - personality_type[4] - personality_type[5] + personality_type[6] - personality_type[7] + personality_type[8] + personality_type[9]
    personality_trait['conscientiousness_score'] = personality_type[0] - personality_type[1] + personality_type[2] -personality_type[3] +personality_type[4] - personality_type[5] +personality_type[6] -personality_type[7] + personality_type[8] + personality_type[9]
    personality_trait['openness_score'] =  personality_type[0] -personality_type[1] + personality_type[2] - personality_type[3] + personality_type[4] - personality_type[5] +personality_type[6] + personality_type[7] + personality_type[8] + personality_type[9]
    
    all_types_scores[name] = personality_trait


# In[74]:


all_types_scores


# In[77]:


all_extroversion = []
all_neuroticism =[]
all_agreeableness =[]
all_conscientiousness =[]
all_openness =[]

for personality_type, personality_trait in all_types_scores.items():
    all_extroversion.append(personality_trait['extroversion_score'])
    all_neuroticism.append(personality_trait['neuroticism_score'])
    all_agreeableness.append(personality_trait['agreeableness_score'])
    all_conscientiousness.append(personality_trait['conscientiousness_score'])
    all_openness.append(personality_trait['openness_score'])


# In[78]:


all_extroversion_normalized = (all_extroversion-min(all_extroversion))/(max(all_extroversion)-min(all_extroversion))
all_neuroticism_normalized = (all_neuroticism-min(all_neuroticism))/(max(all_neuroticism)-min(all_neuroticism))
all_agreeableness_normalized = (all_agreeableness-min(all_agreeableness))/(max(all_agreeableness)-min(all_agreeableness))
all_conscientiousness_normalized = (all_conscientiousness-min(all_conscientiousness))/(max(all_conscientiousness)-min(all_conscientiousness))
all_openness_normalized = (all_openness-min(all_openness))/(max(all_openness)-min(all_openness))


# In[80]:


all_openness_normalized


# In[82]:


counter = 0

normalized_all_types_scores ={}

for personality_type, personality_trait in all_types_scores.items():
    normalized_personality_trait ={}
    normalized_personality_trait['extroversion_score'] = all_extroversion_normalized[counter]
    normalized_personality_trait['neuroticism_score'] = all_neuroticism_normalized[counter]
    normalized_personality_trait['agreeableness_score'] = all_agreeableness_normalized[counter]
    normalized_personality_trait['conscientiousness_score'] = all_conscientiousness_normalized[counter]
    normalized_personality_trait['openness_score'] = all_openness_normalized[counter]
    
    normalized_all_types_scores[personality_type] = normalized_personality_trait
    
    counter+=1


# In[84]:


normalized_all_types_scores


# In[86]:


import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(15,5))
plt.ylim(0, 1)
plt.bar(list(normalized_all_types_scores['one'].keys()), normalized_all_types_scores['one'].values(), color='g')
plt.show()


# In[87]:


plt.figure(figsize=(15,5))
plt.ylim(0, 1)
plt.bar(list(normalized_all_types_scores['two'].keys()), normalized_all_types_scores['two'].values(), color='g')
plt.show()


# In[88]:


plt.figure(figsize=(15,5))
plt.ylim(0, 1)
plt.bar(list(normalized_all_types_scores['three'].keys()), normalized_all_types_scores['three'].values(), color='g')
plt.show()


# In[89]:


plt.figure(figsize=(15,5))
plt.ylim(0, 1)
plt.bar(list(normalized_all_types_scores['four'].keys()), normalized_all_types_scores['four'].values(), color='g')
plt.show()


# In[90]:


plt.figure(figsize=(15,5))
plt.ylim(0, 1)
plt.bar(list(normalized_all_types_scores['five'].keys()), normalized_all_types_scores['five'].values(), color='g')
plt.show()


# In[91]:


plt.figure(figsize=(15,5))
plt.ylim(0, 1)
plt.bar(list(normalized_all_types_scores['six'].keys()), normalized_all_types_scores['six'].values(), color='g')
plt.show()


# In[92]:


plt.figure(figsize=(15,5))
plt.ylim(0, 1)
plt.bar(list(normalized_all_types_scores['seven'].keys()), normalized_all_types_scores['seven'].values(), color='g')
plt.show()


# In[93]:


plt.figure(figsize=(15,5))
plt.ylim(0, 1)
plt.bar(list(normalized_all_types_scores['eight'].keys()), normalized_all_types_scores['eight'].values(), color='g')
plt.show()


# In[94]:


plt.figure(figsize=(15,5))
plt.ylim(0, 1)
plt.bar(list(normalized_all_types_scores['nine'].keys()), normalized_all_types_scores['nine'].values(), color='g')
plt.show()


# In[95]:


plt.figure(figsize=(15,5))
plt.ylim(0, 1)
plt.bar(list(normalized_all_types_scores['ten'].keys()), normalized_all_types_scores['ten'].values(), color='g')
plt.show()

