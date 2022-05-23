import nltk
import uvicorn
from fastapi import FastAPI
import joblib,os


from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import numpy as np

from nltk.tokenize import RegexpTokenizer # regexp tokenizers use to split words from text
from nltk.stem.snowball import SnowballStemmer # stemmes words
from nltk.tokenize import word_tokenize
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
nlp = English()
#tokenizer = Tokenizer(nlp.vocab)

tokenizer = RegexpTokenizer(r'[A-Za-z]+')

black1 =[]
snow_stemmer = SnowballStemmer(language='english')

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

#pkl
# phish_model = open('phishing.pkl','rb')
# phish_model_ls = joblib.load(phish_model)

# ML Aspect

@app.route('/')

@app.route('/home')
def home():
	return render_template('home.html')

@app.route('/mldet')
def web_pre():
    return render_template('detect1.html')

@app.route('/blacklist')
def web_black():
    return render_template('detect2.html')

@app.route('/check')
def web_check():
    return render_template('detect3.html')

@app.route('/predict',methods=['POST'])
def predict1():

	X_predict = []
	features = request.form.values()
	print("featur:",features,type(features))
	X_predict.append(str(features))
	y_Predict = model.predict(features)
	print(y_Predict, features)
	if y_Predict == 'bad':
		ans = "This is a Phishing Site"
		return render_template('detect1.html', result=ans)
	else:
		ans = "This is not a Phishing Site"
		return render_template('detect1.html', result=ans)


@app.route('/addword',methods=['POST'])
def add_word():
	features = request.form.values()

	black1.append(list(features))
	#print(black1,list(features))
	return render_template('detect2.html', result= "Succesfully added")

@app.route('/checklink',methods=['POST'])
def check_list():
	features = request.form.values()
	w1 = list(features)
	token = tokenizer.tokenize(w1[0])
	print(w1)

	print("bl",black1)
	black = [item for sublist in black1 for item in sublist]
	stem_words = []
	for w in token:
		x = snow_stemmer.stem(w)
		stem_words.append(x)
	flag = 0
	for i in range(len(black)):
		if (black[i] in stem_words):
			flag = 1
			print("Element Exists")
			break

	if (flag == 1):
		ans1= "Phishing Website"
		#print("Phishing Website")
	else:
		ans1 = "Legimate Website"
		#print("Legimate Website")
	return render_template('detect3.html', result= ans1)



