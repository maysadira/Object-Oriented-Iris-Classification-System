#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import warnings
warnings.filterwarnings('ignore')


# In[2]:


def load_model(filename):
    """ Load the trained model from a pickle file. """
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model


# In[3]:


def predict_with_model(model, user_input):
    """ Make a prediction using the model and user input. """
    prediction = model.predict([user_input])
    return prediction[0]


# In[ ]:


def main():
    model_filename = 'iris_model.pkl' 
    model = load_model(model_filename)

    user_input = [0.1, 2.5, 5.4, 1.2, 3.3] 
    prediction = predict_with_model(model, user_input)
    print(f"The predicted output is: {prediction}")

if __name__ == "__main__":
    main()


# In[ ]:




