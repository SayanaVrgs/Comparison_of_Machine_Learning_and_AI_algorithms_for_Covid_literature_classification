import pandas as pd
import nltk
from tqdm import tqdm
import re
#from utils import *
import bioc
import glob
from nltk.corpus import stopwords
from string import punctuation


nltk.download('stopwords')

from nltk.corpus import stopwords

stopwords = stopwords.words('english')

 

from string import punctuation

punctuation = list(punctuation)

#function to load assign category.source to litcovid data
def load_all_litcovid(source_file_path = "../data"):

    files = glob.glob(source_file_path+"/litcovid_source*.tsv")

    dfs = []
    #print(files)
    for file in files:
        df = pd.read_csv(file,sep='\t',comment='#')
        #print(file.split('source_')[-1][:-4])
        df["source"] = file.split('source_')[-1][:-4]
        dfs.append(df)
        
    litcovid = pd.concat(dfs)
    
    return litcovid

#Define dataftrame key value pairs
def df_from_dict(d, keys, vals):
    
    df = pd.DataFrame()
    
    df[keys] = list(d.keys())
    df[vals] = list(d.values())
    
    return df

## Loading LitCovid Data Directly

# Deserialize ``fp`` to a BioC collection object.
with open('../data/litcovid2BioCXML.xml','r', encoding='utf-8') as fp:
    collection = bioc.load(fp)

docs = collection.documents

articles = {}
xmls = {}
print("Am here, entering For loop")
for doc in docs:
    try:
        pmid = doc.passages[0].infons['article-id_pmid']
    except:
        pmid = doc.id
        
    text = []
    for passage in doc.passages:
        p = passage.text
        text.append(p)
    
    articles[pmid] = '\n\n'.join(text)

    xmls[pmid] = doc

print("done with For loop")
litcovid_xml_data = df_from_dict(articles,'pmid','text')
#print(litcovid_xml_data)
litcovid_xml_data['length'] = [len(t.split()) for t in litcovid_xml_data['text']]


print("Average Length of full text: " + str(litcovid_xml_data.length.mean()))

litcovid_xml_data['pmid'] = [int(pmid) for pmid in litcovid_xml_data.pmid]

## Using LitCovid Data Directly

litcovid = load_all_litcovid()
litcovid['category'] = litcovid['source']

litcovid_dataset = litcovid_xml_data.merge(litcovid,on='pmid',how='inner')

litcovid_dataset = litcovid_dataset.rename(columns={'title_e': 'title'})

print(litcovid_dataset.columns);
#print(litcovid_dataset)
#Filtering Documents with no text
litcovid_dataset['title_len'] = [len(str(t).split()) for t in litcovid_dataset.title]
litcovid_dataset['diff'] = litcovid_dataset.length - litcovid_dataset.title_len
litcovid_dataset_no_text = litcovid_dataset[litcovid_dataset['diff'] < 25]
litcovid_dataset = litcovid_dataset[litcovid_dataset['diff'] > 25]

print("Number of Articles x Categories in LitCovid: {}".format(len(litcovid_dataset)))

litcovid_dataset['doc_type'] = 'text'
litcovid_dataset.loc[litcovid_dataset['length'] < 300,'doc_type'] = 'abs'

litcovid_dataset = litcovid_dataset[['pmid','text','title','category','length','doc_type']].reset_index(drop=True)
print(litcovid_dataset)

print("Number of Text Articles in LitCovid: {}".format(len(litcovid_dataset.pmid.unique())))

with open('../data/litcovid_category_order.txt','r') as f:
    categories = f.readlines()
    categories = [c.strip() for c in categories]


# Creating Unique Labels

concat_dfs = []

for pmid, df in litcovid_dataset.groupby('pmid'):
    row = df[['pmid','title','text','doc_type']].drop_duplicates()
    
    label = []
    cats = []
    for cat in categories:
        if cat in df.category.values:
            label.append('1')
            cats.append(cat)
        else:
            label.append('0')

    if len(row) > 1:
        i = row.index.values[0]
        row = row[row.index == i]
        
    row['human_label'] = ', '.join(cats)
    row['label'] = ''.join(label)
    
    concat_dfs.append(row)
    
litcovid_dataset_labelled = pd.concat(concat_dfs)
print(litcovid_dataset_labelled)

def tokenize_lines(text):
    
    #stopwords = stopwords.words('english')
    #punctuation = list(punctuation)
    ms = re.finditer('\n+|$',text)

    st = 0

    tokenized_text = []

    for m in ms:
        line = text[st:m.start()]

        sents = nltk.sent_tokenize(line)

        for sent in sents:
            sent = sent.lower()
            words = nltk.word_tokenize(sent)
            cleaned_tokens = [token for token in words if token not in stopwords and token not in punctuation]
            #print(cleaned_tokens)
            tokenized_text.append(cleaned_tokens)
            #print("this text is done")
            
        try:
            if len(tokenized_text) == 0:
                tokenized_text.append([''])
            
            tokenized_text[-1].extend([m.group()])
        except:
            print(tokenized_text)
            raise
            
        st = m.end()

    return tokenized_text

def tokenize_text_w_para(df, column):
    tokenized_pubs = []

    for text in tqdm(df[column]):
        tokenized_text = tokenize_lines(text)
        tokenized_pubs.append(tokenized_text)
    
    return tokenized_pubs

litcovid_dataset_labelled['clean_doc_tokenized'] = tokenize_text_w_para(litcovid_dataset_labelled,'text')
litcovid_dataset_labelled['clean_doc'] = [' '.join([' '.join(sent) for sent in doc]).replace('\n','\\n') for doc in litcovid_dataset_labelled['clean_doc_tokenized']]


litcovid_dataset_labelled['sequence_len'] = [len(t.split()) for t in litcovid_dataset_labelled['text']]
litcovid_dataset_labelled['num_sents'] = [len(t) for t in litcovid_dataset_labelled.clean_doc_tokenized]

print("LitCovid Document Classification Dataset Statistics:")
print("Avg. Sents: {}, Avg. Tokens: {}, 'Total Tokens: {}".format(litcovid_dataset_labelled.num_sents.mean(), litcovid_dataset_labelled.sequence_len.mean(),litcovid_dataset_labelled.sequence_len.sum()))



# ## Splitting Dataset IID Across Different Document Types (Abstract, Full Text and Both)

def split_df_to_train(df, train,val,test):
    df = df.sample(frac=1,random_state=42).reset_index()
    
    train_border = int(train*len(df))
    val_border = int(train*len(df)) + int(val*len(df)) + 1
    
    train_df = df[:train_border]
    val_df = df[train_border:val_border]
    test_df = df[val_border:]
    
    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    return train_df, val_df, test_df

data_split = {'train':[],'val':[],'test':[]}

for doc_type, df in litcovid_dataset_labelled.groupby('doc_type'):
    df.loc[:,'doc_type'] = doc_type
    split = split_df_to_train(df,0.7,0.1,0.2)
    
    data_split['train'].append(split[0])
    data_split['val'].append(split[1])
    data_split['test'].append(split[2])

train = pd.concat(data_split['train'])
val = pd.concat(data_split['val'])
test = pd.concat(data_split['test'])

train[['label','clean_doc']].to_csv('../data/FullLitCovid/train.tsv',header=False,index=False,sep='\t')
val[['label','clean_doc']].to_csv('../data/FullLitCovid/val.tsv',header=False,index=False,sep='\t')
test[['label','clean_doc']].to_csv('../data/FullLitCovid/test.tsv',header=False,index=False,sep='\t')

#Saving Train/Val/Test With All Information

train[['pmid','title','text','clean_doc','clean_doc_tokenized','human_label','label']].to_csv('../data/FullLitCovid/LitCovid.train.csv')
val[['pmid','title','text','clean_doc','clean_doc_tokenized','human_label','label']].to_csv('../data/FullLitCovid/LitCovid.val.csv')
test[['pmid','title','text','clean_doc','clean_doc_tokenized','human_label','label']].to_csv('../data/FullLitCovid/LitCovid.test.csv')


print("Split Statistics:")
print("Train: {}, Val: {}, Test: {}".format(len(train),len(val),len(test)))

print('Done')