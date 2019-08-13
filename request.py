import requests

url = 'http://localhost:5000/api'
url2 = 'http://localhost:5000/popular'
#url3 =  'http://localhost:5000/collaborative'
url4 =  'http://localhost:5000/keras'


#r = requests.post(url,json={'exp':'95c809d493d3657ef06b115fe1e33d3ae1f1df52',}) #passer dans la requete item Id pour recommender des produits similaires 
#print(r.json())

r2 = requests.post(url2,json={'id':'222',}) #passer user ID
print(r2.text)


#r3 = requests.post(url3) 
#print(r3.text)


#r4 = requests.post(url4,json={'id':88,})

#print(r4.text)

