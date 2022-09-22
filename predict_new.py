import pandas as pd
import pickle
from feature_extract import sent2feats

def predict_new(text,model):
    """
    Function to predict the NER tags of a new sentence
    """
    tokens = [[text.split()]]
    tfeats = []
    for sentence in tokens:
        tfeats.append(sent2feats(sentence[0]))
    preds = model.predict(tfeats)
    for texts,pred in zip([text.split()],preds):
        print(texts)
        print(pred)
        pred_df = pd.DataFrame({'words':texts,'ner_tag':pred})
    return pred_df


input_text = str(input("Enter a sentence to tag: "))

# Loading the saved model
loaded_model = pickle.load(open('C:\\Users\\Rajeev\\Python_projects\\NLP\\Customer_NER_training_with NLTK\\model\\finalized_model.sav', 'rb')) 

predict_new(input_text,loaded_model)