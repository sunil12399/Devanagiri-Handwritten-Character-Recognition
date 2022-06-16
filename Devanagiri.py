
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import models, layers


# In[2]:


from sklearn.model_selection import train_test_split


# In[3]:


IMAGE_SIZE = 32
BATCH_SIZE = 64
EPOCH = 15


# In[4]:


df = pd.read_csv("data.csv", usecols=lambda x: x != 'character', dtype = 'uint8')
df['character'] = pd.read_csv("data.csv", usecols=['character'])
df.head()


# In[5]:


df.info()


# In[6]:


df.shape


# In[7]:


np.sum(df.isna().sum()>0)


# In[8]:


characters = df.character.unique()
characters


# In[9]:


df.character.value_counts()

X_images = df.iloc[:, :-1].values.reshape((-1,32,32))
fig, ax = plt.subplots(1,4, figsize = (12,9))

for i, j in enumerate(np.random.randint(92000, size=(4))):
    ax[i].imshow(X_images[j])
    ax[i].title.set_text(df.iloc[j, -1])

plt.show()

# In[10]:


X, y = df.drop('character', axis = 1), df.character


# In[11]:


train_X, test_X, train_y, test_y = train_test_split(X, y)


# In[12]:


from sklearn.preprocessing import LabelBinarizer
labelbinarize = LabelBinarizer()
train_y = labelbinarize.fit_transform(train_y)


# In[13]:


train_X = np.divide(train_X, 255)
test_X = np.divide(test_X, 255)


# In[14]:


train_X = train_X.values.reshape(-1, 32, 32, 1)
test_X = test_X.values.reshape(-1, 32, 32, 1)


# In[15]:


print(X.shape)
print(train_X.shape)
print(test_X.shape)
print(train_y.shape)
print(test_y.shape)


# In[16]:


num_classes = train_y.shape[1]
input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)

model = tf.keras.Sequential([
        
  layers.Conv2D(128, (4,4), activation='relu', input_shape=input_shape),
  layers.MaxPooling2D((2,2)),
  layers.Dropout(0.2),
        
  layers.Conv2D(32,(3,3), activation='relu'),
  layers.MaxPooling2D((2,2)),
  layers.Dropout(0.2),
        
  layers.Conv2D(32, (2,2), activation='relu'),
  layers.MaxPooling2D((2,2)),
  layers.Dropout(0.2),
        
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, activation='softmax'),
])

model.build(input_shape=input_shape)


# In[17]:


model.summary()


# In[18]:


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics = ['accuracy']
)


# In[19]:


from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    min_delta=0.05, 
    patience = 2, 
    restore_best_weights=True,
)


# In[20]:


history = model.fit(
    train_X,
    train_y,
    epochs = EPOCH,
    batch_size = BATCH_SIZE,
    verbose = 1,
    validation_split=0.2,
    callbacks = [early_stopping]
)


# In[21]:


history.history.keys()


# In[24]:


from operator import pos
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))


# In[26]:


plt.figure(
  figsize=(12,4)
)
plt.subplot(1,2,1)
plt.plot(epochs, acc, label = 'Training Accuracy')
plt.plot(epochs, val_acc, label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
_ = plt.title("Training Vs Validation Accuracy")


# In[29]:


plt.figure(
  figsize=(12,4)
)
plt.subplot(1,2,1)
plt.plot(epochs, loss, label = 'Training Loss')
plt.plot(epochs, val_loss, label = 'Validation Loss')
plt.legend(loc = 'upper right')
_ = plt.title("Training Vs Validation Loss")


# In[30]:


predictions = model.predict(test_X)
predictions.shape


# In[31]:


preds = labelbinarize.inverse_transform(predictions)


# In[32]:


preds.shape


# In[33]:


from sklearn.metrics import classification_report, f1_score, ConfusionMatrixDisplay


# In[34]:


print(classification_report(test_y, preds))


# In[35]:


f1_score(test_y, preds, average = 'weighted')

