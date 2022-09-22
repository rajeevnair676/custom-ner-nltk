from load_data import load_conll_transform,data_transform
from feature_extract import get_feats_conll
from model import train_seq


def main(): 
    try:
        PATH = 'c:\\Users\\Rajeev\\Python_projects\\NLP\\Customer_NER_training_with NLTK\\data\\payment_terms.jsonl'
        train_data,test_data = load_conll_transform(PATH,split=20) 
    except:
        print("Correct the PATH variable with the required path to the file")

    conll_train = data_transform(train_data)
    conll_dev = data_transform(test_data)
    
    print("Training a Sequence classification model with CRF")
    feats, labels = get_feats_conll(conll_train)
    devfeats, devlabels = get_feats_conll(conll_dev)

    model_name = 'C:\\Users\\Rajeev\\Python_projects\\NLP\\Customer_NER_training_with NLTK\\model\\finalized_model.sav'
    train_seq(feats, labels, devfeats, devlabels,model_name)

if __name__=="__main__":
    main()