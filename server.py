# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from keras.models import model_from_json

app = Flask(__name__)

# Load the model
model = pickle.load(open('modelContentBased.pkl','rb'))
model2 = pickle.load(open('modelCollaborativeBased.pkl','rb'))
#modelKeras = pickle.load(open('modelKERAS.pkl','rb'))


dataset =pd.read_csv('generatedDataForTesting/dataset.csv')




#Collaborative filtering - user based- neural network  - RBM with tensorflow - 
#@app.route('/collaborative',methods=['POST'])
#def collaborativeFiltering():
#    return jsonify(model2)
    


#Collaborative filtering - neural network - keras 
@app.route('/keras',methods=['POST'])
def collaboratviveKeras():
    data = request.get_json(force=True)
    df_interaction =pd.read_csv('generatedDataForTesting/df_interaction.csv')


    # load json and create model
    json_file = open('modelKeras.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("modelKeras.h5")
    print("Loaded model from disk")
    #get the user id correspondant
    for i in range(len(df_interaction)):
        if df_interaction['userId'][i]==data['id']:
            idu=df_interaction['userId_encoded'][i]
            break

    
    # Creating dataset for making recommendations for the a given user
    product_data = np.array(list(set(df_interaction.itemId_encoded)))
    user = np.array([idu for i in range(len(product_data))])
    predictions = loaded_model.predict([user, product_data])
    predictions = np.array([a[0] for a in predictions])
    recommended_product_ids = (-predictions).argsort()[:5]
 
    recDf= pd.DataFrame(recommended_product_ids)
    recDf.columns = ['itemId_encoded']
    recommended = recDf.merge(df_interaction, on='itemId_encoded')
    del df_interaction['Unnamed: 0']
    del recommended['Unnamed: 0']

    del recommended['userId']
    del recommended['numSearch']
    del recommended['userId_encoded']
    recommended = recommended.drop_duplicates()
    recommended['prediction'] = predictions[recommended_product_ids]

    print(recommended)


    return jsonify(recommended.values.tolist())








#Popularity based to show products that have been searched the most by users #les plus recherchés
@app.route('/popular',methods=['POST'])
def popularityBasedRecommendation():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    df_interaction =pd.read_csv('generatedDataForTesting/searchuseritem.csv')
    df_interaction = df_interaction.groupby(['itemId','userId']).size().reset_index()
    df_interaction.rename(columns = {0: 'numSearch'}, inplace = True)
    df_interaction["numSearch"].max()

    max_search = df_interaction["numSearch"].max() # the max frequency of search = 2 
    top_searched = df_interaction[df_interaction["numSearch"] == max_search]  


    # To get the most popular products among the users
    pop_products_id = list(top_searched.groupby('itemId').count()["numSearch"].sort_values(ascending=False)[0:500].index)

    # The products that are searched by user id
    products_id_searched_user = list(df_interaction[df_interaction["userId"] == int(data['id'])].itemId)  #exp : user id 222

    # The popular products list that are not watched by user id
    rec_list = np.setdiff1d(pop_products_id, products_id_searched_user)

    recommendedProd = pd.DataFrame(rec_list)

    recommendedProd.columns = ['itemId']

    recommendedProd = dataset.merge(recommendedProd, on='itemId')
    t=[]
    recommendedProd['itemId']
    for i in range(len(recommendedProd['itemId'])):
    	t.append(recommendedProd['itemId'][i])
    
    return jsonify(t)






@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)

    cosine_sim = cosine_similarity(model, model)
    idProducts = dataset['itemId']
    indices = pd.Series(dataset.index, index=dataset['itemId'])

    idx = indices[data['exp']]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    product_indices = [i[0] for i in sim_scores]
    d= idProducts.iloc[product_indices]
    output = d.head(5)
    output = pd.DataFrame(output)
    output= output.reset_index()
    t=[]
    for i in range(len(output['itemId'])): 
    	t.append(output['itemId'][i])
    

    return jsonify(t)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
