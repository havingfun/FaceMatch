# FaceMatch
Streamlit App for Face Matching fun - Using a Bollywood Celebrity Model

Checkout Demo [here](http://ml.newsforaction.in)

# Model Training - 
You can either use a [pretrained model](https://www.kaggle.com/havingfun/bollywood-celeb-face-recognizer-model) for around 100 celebs that I trained using this [dataset](https://www.kaggle.com/havingfun/100-bollywood-celebrity-faces)
Or you can train your own dataset using the following architecture [Face Recognizer by ArsFutura](https://github.com/arsfutura/face-recognition)

# How to Run
Once you have the model ready, you have to create model folder and move your model to that folder.
```
mkdir model & cd model
cp your_model .
```
Then run streamlit app
```
streamlit run app.py
```
