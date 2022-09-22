
import jsonlines
import spacy


def load_conll_transform(path,split=20):
    """
    Function to load the data from the prescribed path, and transform it to conll format, and split to train and test sets
    """
    data=[]
    with jsonlines.open(path) as f:
          for line in f.iter():
              if line['labels']:
                 data.append((line['text'],{'entities':line['labels']}))
    nlp = spacy.blank("en")
    data_conll=[]
    for text, labels in data:
        doc = nlp(text)
        ents = []
        tok_conll=[]
        for start, end, label in labels["entities"]:
            ents.append(doc.char_span(start, end, label))
        doc.ents = ents
        for tok in doc:
            label = tok.ent_iob_
            if tok.ent_iob_ != "O":
               label += '-' + tok.ent_type_
            tok_conll.append([str(tok),label])
        data_conll.append(tok_conll)
    train_data = data_conll[:split]
    test_data = data_conll[split:]
    return train_data,test_data


def data_transform(data_conll):
    """
    Function to transform the data into word, tags format
    """
    input_data = []
    for sent in data_conll:
        words=[]
        tags=[]
        for word,tag in sent:
            words.append(word)
            tags.append(tag)
        input_data.append([words,tags])
    return input_data