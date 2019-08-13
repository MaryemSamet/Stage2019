# Importing the libraries
import pickle
import requests
import json
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn import preprocessing
import tensorflow as tf
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from keras.layers import Input, Embedding, Flatten, Dot, Dense
from keras.models import Model






def dataProcessing(products_df):
    ##data preprocessing 
    df_Jumia = products_df[products_df['site'] == 'jumia']
    df_Tunisianet = products_df[products_df['site'] == 'tunisiaNet']
    df_MyTek = products_df[products_df['site'] == 'MyTek']
    df_MegaTn = products_df[products_df['site'] == 'Mega.tn']

    df_Jumia=df_Jumia.reset_index()
    df_Tunisianet=df_Tunisianet.reset_index()
    df_MyTek=df_MyTek.reset_index()
    df_MegaTn=df_MegaTn.reset_index()
    return df_Jumia , df_Tunisianet , df_MyTek , df_MegaTn






def jumiaDataProcessing(df_Jumia):
    #Fist Case : site jumia ; with 5 features ;  
    splitted = []
    for i in range(len(df_Jumia['description'])):
        if df_Jumia['description'][i].count('-') == 5:
            line =[]
            line.append(df_Jumia['_id'][i])
            line.append(df_Jumia['description'][i].split('-'))
            splitted.append(line)
            
    
    # we've got a dataset with different product features  , but there are some messy columns with wrong values 
    #for example line 7 
    #we have to choose particular features to save for all products  ,
    # name #memoire ram # memoire # couleur # garantie 


    lines = []
    dataSplitted=pd.DataFrame(splitted)
    r = re.compile('.*Go|.*Mo')
    for j in range(len(dataSplitted[1])):

        if r.match(dataSplitted[1][j][1]) is not None:
            line=[]
            line.append(dataSplitted[0][j])
            line.append(dataSplitted[1][j][0])
            line.append(dataSplitted[1][j][1])
            line.append(dataSplitted[1][j][2])
            line.append(dataSplitted[1][j][4])
            line.append(dataSplitted[1][j][5])

            lines.append(line)
        else:
            line=[]
        
            line.append(dataSplitted[0][j])
            line.append(dataSplitted[1][j][0])
            line.append(dataSplitted[1][j][2])
            line.append(dataSplitted[1][j][3])
            line.append(dataSplitted[1][j][4])
            line.append(dataSplitted[1][j][5])


            lines.append(line)


    features1 = pd.DataFrame(lines)

    ##CASE 1 done ##

    features1.columns = ['_id','name','ram','memory','color','guarantee']
    
    
    
    

    #Second Case : site jumia ; with 6 features ;  
    splitted = []
    for i in range(len(df_Jumia['description'])):
        if df_Jumia['description'][i].count('-') == 6:
            line =[]
            line.append(df_Jumia['_id'][i])
            line.append(df_Jumia['description'][i].split('-'))
            splitted.append(line)
    
    

    lines = []
    r = re.compile('.*".*')
    r2 = re.compile('.*GO.*') 
    r3 = re.compile('.*1 An.*|.*1 an.*')   
    r4 = re.compile('.*mAh.*')   




    dataSplitted=pd.DataFrame(splitted)
    for j in range(len(dataSplitted[1])):
        if r.match(dataSplitted[1][j][2]) is not None:
            line = []
        
        
            line.append(dataSplitted[0][j])
            line.append(dataSplitted[1][j][0]+" "+ dataSplitted[1][j][1])
            line.append(dataSplitted[1][j][3])
            line.append(dataSplitted[1][j][4])
            line.append(dataSplitted[1][j][5])
            line.append(dataSplitted[1][j][6])
        
        
            lines.append(line)
        else:
            if r2.match(dataSplitted[1][j][5]) is not None:
            
            
            
                line=[]
            
            
                    
                line.append(dataSplitted[0][j])
                line.append(dataSplitted[1][j][0])
                line.append(dataSplitted[1][j][2])
                line.append(dataSplitted[1][j][3])
                line.append(dataSplitted[1][j][4])
                line.append(dataSplitted[1][j][6])
        
        
                lines.append(line)
            elif r3.match(dataSplitted[1][j][5]) is not None:
                line=[]
                line.append(dataSplitted[0][j])
                line.append(dataSplitted[1][j][0])
                line.append(dataSplitted[1][j][2])
                line.append(dataSplitted[1][j][3])
                line.append(dataSplitted[1][j][4])
                line.append(dataSplitted[1][j][5])
    

                lines.append(line)
            elif r4.match(dataSplitted[1][j][5]) is not None:
                line=[]
            
                line.append(dataSplitted[0][j])
                line.append(dataSplitted[1][j][0])
                line.append(dataSplitted[1][j][2])
                line.append(dataSplitted[1][j][3])
                line.append(dataSplitted[1][j][4])
                line.append(dataSplitted[1][j][6])
    

                lines.append(line)
            
            elif r.match(dataSplitted[1][j][3]) is not None:
                line=[]
                line.append(dataSplitted[0][j])
                line.append(dataSplitted[1][j][0])
                line.append(dataSplitted[1][j][1])
                line.append(dataSplitted[1][j][2])
                line.append(dataSplitted[1][j][5])
                line.append(dataSplitted[1][j][5])
        
    

                lines.append(line)
            
            
            
            else:
        
                line=[]
                line.append(dataSplitted[0][j])
                line.append(dataSplitted[1][j][0])
                line.append(dataSplitted[1][j][2])
                line.append(dataSplitted[1][j][3])
                line.append(dataSplitted[1][j][5])
                line.append(dataSplitted[1][j][6])
    

                lines.append(line)


    features2 = pd.DataFrame(lines)

    ##CASE 2 done ##
    features2.columns = ['_id','name','ram','memory','color','guarantee']
    
    
    
    
    
    #3rd Case : site jumia ; with 4 features ;  
    splitted = []
    for i in range(len(df_Jumia['description'])):
        if df_Jumia['description'][i].count('-') == 4:
            line =[]
            line.append(df_Jumia['_id'][i])
            line.append(df_Jumia['description'][i].split('-'))
            splitted.append(line)
        
        

    lines = []
    dataSplitted=pd.DataFrame(splitted)
    r = re.compile('.*Go.*|.*Mo.*')
    r1 = re.compile('.*".*')

    for j in range(len(dataSplitted[1])):

        if r.match(dataSplitted[1][j][1]) is None:
            if r.match(dataSplitted[1][j][2]) is not None:
                line=[]
                line.append(dataSplitted[0][j])
                line.append(dataSplitted[1][j][0])
                line.append(dataSplitted[1][j][2])
                line.append(dataSplitted[1][j][3])
                line.append("")
                line.append(dataSplitted[1][j][4])

                lines.append(line)
            else:
            
            
                line=[]
                line.append(dataSplitted[0][j])
                line.append(dataSplitted[1][j][0]+" "+dataSplitted[1][j][1])
                line.append("")
                line.append("")
                line.append("")
                line.append(dataSplitted[1][j][4])

                lines.append(line)
        
        else:
            line=[]
            line.append(dataSplitted[0][j])
            line.append(dataSplitted[1][j][0])
            line.append(dataSplitted[1][j][1])
            line.append(dataSplitted[1][j][2])
            line.append(dataSplitted[1][j][3])
            line.append(dataSplitted[1][j][4])
    

            lines.append(line)


    features3 = pd.DataFrame(lines)

    ##CASE 3 done ##
    features3.columns = ['_id','name','ram','memory','color','guarantee']


    frames = [features1, features2, features3]

    result = pd.concat(frames)
    jumiaProducts = products_df.merge(result, on='_id')
    #### DataFrame for Jumia is reaaaaaaaaaaaadyyyyy :D 

    jumiaProducts.to_csv(r'generatedDataForTesting/jumiaProducts.csv')
    
    
    return jumiaProducts



def mytekDataProcessing(df_MyTek):
    ## Now we will treat MyTek products 
    #the problem in this case is : the color is in the title column 
    #the name of product is concatenated with the color 
    #we will get from the description only the memory and ram 


    colors = ["Bleu", "Noir", "Gold","Gris","Brown","Blanc","Lavender","Brun ambré","Rouge","Black","Vert"," Iris Violet","Rose","Gray","Bleu Saphir","Aurora purple","noir","SpaceGray","Space Grey","Breathing Crystal","Silver","Violet","Vert Emeraude","Aurora Green","Brun","White","Blue","Gris sombre","Bleu Corail","Gris Sidéral","Midnight Black","Moka Brown","jewelry white","Violet foncé","Thunder noir","Peacock Bleu","Blue Copper","Bleu Saphir","Violet","NOIR","BLEU","Purple","Grey","Helium","Brune","Champagne Gold","Iron Gris","Aqua Bleu","Marron","Brun ambré","Cuivre","Nacré","Aurora","Cristal","Corail","Titanium Grey","Noir Carbone","Steel","Violet Foncé","Gris Foncé","Baltique","White","Bleu Ciel"]

    lines=[]
    for i in range(len(df_MyTek)):
    
        line=[]
        line.append(df_MyTek['_id'][i])
    
        word_list = df_MyTek['title'][i].split()  


        line.append(word_list[1]+" "+ word_list[2]+" "+word_list[3]+" "+word_list[4])

    
        text = df_MyTek['description'][i]
    
        m = re.search('RAM:(.+?)-', text)
        if m:
            ram = m.group(1)
    

        m2 = re.search('Stockage:(.+?)-', text)
        if m2:
            memory = m2.group(1)
    
        line.append(ram)
        line.append(memory)
    
   

        for color in colors:
            if color in  df_MyTek['title'][i]:
                line.append(color)
                break

    
    
        lines.append(line)

    dfMytek = pd.DataFrame(lines)    

    dfMytek.columns = ['_id','name','ram','memory','color']

    mytekProducts = products_df.merge(dfMytek, on='_id')

    mytekProducts.to_csv(r'generatedDataForTesting/mytekProducts.csv')
    ## DATAFRAME for mytek is readyyyyyyyyyyyyyyyy
    
    return mytekProducts



def tunisianetDataProcessing(df_Tunisianet):
    ## Now we will treat TunisiaNet products 

    lines=[]
    for i in range(len(df_Tunisianet)):
    
        line=[]
        line.append(df_Tunisianet['_id'][i])
        text1 = df_Tunisianet['title'][i]

    
        m4 = re.search('(.+?)\/', text1)
        if m4:
            name = m4.group(1)
    
    
        text = df_Tunisianet['description'][i]
    
        m = re.search('RAM (.+?)-', text)
        if m:
            ram = m.group(1)
    

        m2 = re.search('Mémoire(.+?)-', text)
        if m2:
            memory = m2.group(1)
        
        
    
   

        m3 = re.search('Couleur(.+?)-', text)
        if m3:
            color = m3.group(1)
        
        
        line.append(name)
    
        line.append(ram)
        line.append(memory)
        line.append(color)

    
    
        lines.append(line)

    df_Tunisianet = pd.DataFrame(lines)    

    df_Tunisianet.columns = ['_id','name','ram','memory','color']


    tunisianetProducts = products_df.merge(df_Tunisianet, on='_id')
    tunisianetProducts.to_csv(r'generatedDataForTesting/tunisianetProducts.csv')
    ## DATAFRAME for tunisianet  is readyyyyyyyyyyyyyyyy
    return tunisianetProducts




def megatnDataProcessing(df_MegaTn):
    ## Now we will treat MegaTn products 

    lines=[]
    for i in range(len(df_MegaTn)):
    
        line=[]
        line.append(df_MegaTn['_id'][i])
        line.append(df_MegaTn['title'][i])

    
        text = df_MegaTn['description'][i]
        m = re.search(r'RAM ?:?(.+?)(-|,|Mémoire)', text)
        if m:
            ram = m.group(1)
        
        m2 = re.search(r'Stockage ?:?(.+?)(-|,)', text)
        m3 = re.search(r'Mémoire ?:?(.+?)Go', text)
        m4 = re.search('Couleur ?:?(.+?)"', text)

        if m2:
            memory = m2.group(1)
        elif m3:
            memory = m3.group(1)+"Go"
        if m4:
            color = m4.group(1)
        else:
            color=''
             
        
    
        line.append(ram)
        line.append(memory)
        line.append(color)

    
    
        lines.append(line)

    MegaTnDf = pd.DataFrame(lines)    

    MegaTnDf.columns = ['_id','name','ram','memory','color']

    #delete messy rows : those who don't have a specefic unit for example : means that it hasn't been tottaly extracted 

    MegaTnDf=MegaTnDf[MegaTnDf['ram'].str.contains('Go|Mo', na=False)]

    #Clean columns
    MegaTnDf['color']=MegaTnDf['color'].str.replace('-', '')
    MegaTnDf['color']=MegaTnDf['color'].str.replace(',', '')
    MegaTnDf['memory']=MegaTnDf['memory'].str.replace(':', '')
    MegaTnDf['memory']=MegaTnDf['memory'].str.replace(',', '')


    megaTnProducts = products_df.merge(MegaTnDf, on='_id')
    megaTnProducts.to_csv(r'generatedDataForTesting/megaTnProducts.csv')

    ## DATAFRAME for megaTn  is readyyyyyyyyyyyyyyyy
    return megaTnProducts



def concatenateDatasets(megaTnProducts,tunisianetProducts,mytekProducts,jumiaProducts):
    dataset = jumiaProducts.append(mytekProducts, ignore_index = True).append(megaTnProducts, ignore_index = True).append(tunisianetProducts, ignore_index = True)
    dataset.to_csv(r'generatedDataForTesting/dataset.csv')
    return dataset


def cleanDataset(dataset):
    
    #delete useless columns
    del dataset['_index']

    del dataset['_score']

    del dataset['_type']
    del dataset['href']
    del dataset['image']
    del dataset['title']
    del dataset['description']


    #get price only 
    dataset['price'] = dataset['price'].map(lambda x: re.sub(r'\W+', '', x))
    dataset['price'] = dataset['price'].str.extract('(\d+)').astype(float)

    #replace Nan with empty string 
    dataset = dataset.replace(np.nan, '', regex=True)

    dataset.reset_index(drop=True)


    dataset['name']=dataset['name'].str.replace('"', ' ')
    dataset['name']=dataset['name'].str.replace(']', '')
    dataset['name']=dataset['name'].str.replace('[', '')
    dataset['guarantee']=dataset['guarantee'].str.replace(']', '')
    dataset['guarantee']=dataset['guarantee'].str.replace('"', '')

    dataset.rename(columns={'_id':'itemId'}, inplace=True)

    dataset.to_csv(r'generatedDataForTesting/ProductsdatasetCleaned.csv')


    dataset['name'] = dataset['name'].astype('str').apply(lambda x: str.lower(x.replace("-", " ")))
    dataset['ram'] = dataset['ram'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
    dataset['memory'] = dataset['memory'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
    dataset['guarantee'] = dataset['guarantee'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
    dataset['color'] = dataset['color'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))



    dataset['soup'] =  dataset['ram']+ ' ' + dataset['memory']+ ' ' +dataset['color']
    dataset.to_csv(r'generatedDataForTesting/dataset.csv')
    
    return dataset


#CountVectorizer for content based recommendation system 
def contentBasedFiltering(dataset):
    
    count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0)
    count_matrix = count.fit_transform(dataset['soup'])
    
    return count_matrix





#Collaborative based filtering - user based - KERAS - neural network 

def collaborativeKERAS(df_interaction):

    df_interaction = df_interaction.groupby(['itemId','userId']).size().reset_index()
    df_interaction.rename(columns = {0: 'numSearch'}, inplace = True)
    x_array = np.array(df_interaction['numSearch'])
    normalized_X = preprocessing.normalize([x_array])
    df_interaction['numSearch']=normalized_X.tolist()[0]
    #change itemId to categoric 
    df_interaction['itemId_encoded'] = df_interaction.itemId.astype('category').cat.codes
    df_interaction['userId_encoded'] = df_interaction.userId.astype('category').cat.codes
    train, test = train_test_split(df_interaction, test_size=0.2, random_state=42)
    n_users = len(df_interaction.userId_encoded.unique())
    n_products = len(df_interaction.itemId_encoded.unique())
    product_input = Input(shape=[1], name="Product-Input")
    product_embedding = Embedding(n_products+1, 5, name="Product-Embedding")(product_input)
    product_vec = Flatten(name="Flatten-products")(product_embedding)
    user_input = Input(shape=[1], name="User-Input")
    user_embedding = Embedding(n_users+1, 5, name="User-Embedding")(user_input)
    user_vec = Flatten(name="Flatten-Users")(user_embedding)
    prod = Dot(name="Dot-Product", axes=1)([product_vec, user_vec])
    model = Model([user_input, product_input], prod)
    model.compile('adam', 'mean_squared_error')
    history = model.fit([train.userId_encoded, train.itemId_encoded], train.numSearch, epochs=10, verbose=1)
    df_interaction.to_csv(r'generatedDataForTesting/df_interaction.csv')

    return model









#Collaborative based filtering - user based - with RBM - tensorflow  
def collaborativeWithRBM(df_product,df_interaction):

    #delete useless columns
    del df_product['_index']
    del df_product['_score']
    del df_product['_type']
    del df_product['href']
    del df_product['image']
  

    #delete nan rows 
    df_product.dropna()
    df_product.rename(columns={'_id':'itemId'}, inplace=True)

    #get dataset that represent the search behaviour of users
    #each rows represents a search - item 6a84eeb74e0e1ca2e79d7c2ae8c952d65cc9c5be  appeared in search of user 2 -


    df_interaction = df_interaction.groupby(['itemId','userId']).size().reset_index()
    df_interaction.rename(columns = {0: 'numSearch'}, inplace = True)
   

    x_array = np.array(df_interaction['numSearch'])
    normalized_X = preprocessing.normalize([x_array])
    df_interaction['numSearch']=normalized_X.tolist()[0]
    df_product['List Index'] = df_product.index
    merged_df = df_product.merge(df_interaction, on='itemId')
    merged_df = merged_df.dropna()
    # Drop unnecessary columns
    merged_df = merged_df.drop('price', axis=1).drop('title', axis=1).drop('site', axis=1).drop('description',axis=1)

    user_Group = merged_df.groupby('userId')


    # Amount of users used for training
    amountOfUsedUsers = len(user_Group)-1
    # Creating the training list
    trX = []

    # For each user in the group
    for userID, curUser in user_Group:

        # Create a temp that stores every product's frequency search
        temp = [0]*len(df_product)

        # For each movie in curUser's movie list
        for num, product in curUser.iterrows():

            # Divide the rating by 5 and store it
            temp[product['List Index']] = product['numSearch']

        # Add the list of frequency search into the training list
        trX.append(temp)

        # Check to see if we finished adding in the amount of users for training
        if amountOfUsedUsers == 0:
            break
        amountOfUsedUsers -= 1
    # Setting the models Parameters
    hiddenUnits = 10
    visibleUnits = len(df_product)
    vb = tf.placeholder(tf.float32, [visibleUnits])  # Number of unique products
    hb = tf.placeholder(tf.float32, [hiddenUnits])  # Number of features were going to learn
    W = tf.placeholder(tf.float32, [visibleUnits, hiddenUnits])  # Weight Matrix
    # Phase 1: Input Processing
    v0 = tf.placeholder("float", [None, visibleUnits])
    _h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)  # Visible layer activation
    h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))  # Gibb's Sampling

    # Phase 2: Reconstruction
    _v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb)  # Hidden layer activation
    v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
    h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)
    """ Set RBM Training Parameters """

    # Learning rate
    alpha = 1.0

    # Create the gradients
    w_pos_grad = tf.matmul(tf.transpose(v0), h0)
    w_neg_grad = tf.matmul(tf.transpose(v1), h1)

    # Calculate the Contrastive Divergence to maximize
    CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(v0)[0])

    # Create methods to update the weights and biases
    update_w = W + alpha * CD
    update_vb = vb + alpha * tf.reduce_mean(v0 - v1, 0)
    update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)

    # Set the error function, here we use Mean Absolute Error Function
    err = v0 - v1
    err_sum = tf.reduce_mean(err*err)
    """ Initialize our Variables with Zeroes using Numpy Library """

    # Current weight
    cur_w = np.zeros([visibleUnits, hiddenUnits], np.float32)

    # Current visible unit biases
    cur_vb = np.zeros([visibleUnits], np.float32)

    # Current hidden unit biases
    cur_hb = np.zeros([hiddenUnits], np.float32)

    # Previous weight
    prv_w = np.zeros([visibleUnits, hiddenUnits], np.float32)

    # Previous visible unit biases
    prv_vb = np.zeros([visibleUnits], np.float32)

    # Previous hidden unit biases
    prv_hb = np.zeros([hiddenUnits], np.float32)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # Train RBM with 15 Epochs, with Each Epoch using 10 batches with size 100, After training print out the error by epoch
    epochs = 15
    batchsize = 100
    errors = []
    for i in range(epochs):
        for start, end in zip(range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
            batch = trX[start:end]
            cur_w = sess.run(update_w, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
            cur_vb = sess.run(update_vb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
            cur_hb = sess.run(update_hb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
            prv_w = cur_w
            prv_vb = cur_vb
            prv_hb = cur_hb
        errors.append(sess.run(err_sum, feed_dict={v0: trX, W: cur_w, vb: cur_vb, hb: cur_hb}))
        print(errors[-1])
    plt.plot(errors)
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.show()
    
   

  


    """
    Recommendation System :-
    - We can now predict products that an arbitrarily selected user might like. 
    - This can be accomplished by feeding in the user's watched movie preferences into the RBM and then reconstructing the 
      input. 
    - The values that the RBM gives us will attempt to estimate the user's preferences for movies that he hasn't watched 
      based on the preferences of the users that the RBM was trained on.
    """

    #get the indice row of the user id # we want to get the array that represent the user transactions
    #for i in range(len(merged_df)):
     #   if merged_df.iloc[i][2] == idUSER:
      #      break

    recommendationForAllUsers=[]
    for i in range(len(trX)):
        userID = merged_df.iloc[i][2]
        # Select the input User
        inputUser = [trX[i]]
        # Feeding in the User and Reconstructing the input
        hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
        vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
        feed = sess.run(hh0, feed_dict={v0: inputUser, W: prv_w, hb: prv_hb})
        rec = sess.run(vv1, feed_dict={hh0: feed, W: prv_w, vb: prv_vb})
        #List the 4 most recommended products for our mock user by sorting it by their scores given by our model.
        scored_products_df_4 = df_product
        scored_products_df_4["Recommendation Score"] = rec[0]
        scored_products_df_4.sort_values(["Recommendation Score"], ascending=False).head()
        """ Recommend User what products he has not watched yet """

        # Find all products the mock user has watched before
        products_df_4 = merged_df[merged_df['userId'] == merged_df.iloc[i][2]]
        products_df_4.head()
        """ Merge all products that our mock users has watched with predicted scores based on his historical data: """
        # Merging df_product with interaction_df by itemId
        merged_df_4 = scored_products_df_4.merge(products_df_4, on='itemId', how='outer')
        # Dropping unnecessary columns
        merged_df_4 = merged_df_4.drop('List Index_y', axis=1).drop('userId', axis=1)
        # Sort and take a look at first 2 rows
        line=[]
        line.append(userID)
        line.append(merged_df_4.sort_values(['Recommendation Score'], ascending=False).head())
        recommendationForAllUsers.append(line)
    #we return all recommended products for every user 

    return recommendationForAllUsers















#For collaborative filtering - KERAS - neural network 

products_df =pd.read_csv('generatedDataForTesting/Products.csv')
df_interaction =pd.read_csv('generatedDataForTesting/searchuseritem.csv')


modelKeras = collaborativeKERAS(df_interaction)



#For Collaborative filtering - RBM - Tensorflow -

#df_interaction =pd.read_csv('generatedDataForTesting/searchuseritem.csv')
#df_product =pd.read_csv('generatedDataForTesting/Products.csv')

#recommendationForAllUsers = collaborativeWithRBM(df_product,df_interaction)
#pickle.dump(recommendationForAllUsers, open('modelCollaborativeBased.pkl','wb'))



#Preprocessing of our data # cleaning # treatment  # before applying to content based model
products_df =pd.read_csv('generatedDataForTesting/Products.csv')
df_Jumia , df_Tunisianet , df_MyTek , df_MegaTn = dataProcessing(products_df)
jumiaProducts = jumiaDataProcessing(df_Jumia)
mytekProducts = mytekDataProcessing(df_MyTek)
tunisianetProducts = tunisianetDataProcessing(df_Tunisianet)
megaTnProducts = megatnDataProcessing(df_MegaTn)
dataset = concatenateDatasets(megaTnProducts,tunisianetProducts,mytekProducts,jumiaProducts)
dataset = cleanDataset(dataset)


#For Content based , we used countVectorize
count_matrix = contentBasedFiltering(dataset)




# Saving model to disk
pickle.dump(count_matrix, open('modelContentBased.pkl','wb'))
pickle.dump(modelKeras, open('modelKERAS.pkl','wb'))

# serialize model to JSON
model_json = modelKeras.to_json()
with open("modelKeras.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
modelKeras.save_weights("modelKeras.h5")
print("Saved model to disk")





