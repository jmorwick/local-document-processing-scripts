from io import BytesIO
import cidnilib
import numpy as np
from cidnilib import FileBasedDataService as FBDS
from cidnilib import PickleFileBasedDataService as PFBDS
from cidnilib import InMemoryKnowledgeService as IMKS
import PIL
import json
import sys


if len(sys.argv) < 4:
   print("usage: python script pdf_list_cid ocr_model embedding_model")

pdf_list_cid = sys.argv[1]
ocr_model_name = sys.argv[2]
embedding_model_name = sys.argv[3]


print('using embeddings from',embedding_model_name,'derived from OCR from',ocr_model_name)



ds = FBDS('./cidnidb')
kds = PFBDS('./cidnidb', levels=0)
ks = IMKS(kds)


document_cids = []



for line in ds.recall(ds.decode(pdf_list_cid)).decode().split('\n'):
        line = line.strip() 
        if not line: continue
        cid = ds.decode(line)   
        document_cids.append(cid)
        
def get_property(cid, property):
    if type(cid) == bytes: cid = ds.encode(cid) 
    try: return next(ks.inquire(cid, property))[2]
    except: return None
        

def get_page_embeddings(cid, page_num=None):
    embeddings = []
    for _, _, page_cid in ks.inquire(ds.encode(cid), 'CONTAINS'):
        kid, _ = ks.believe(ds.encode(cid), 'CONTAINS', page_cid)
        if page_num and get_property(kid, 'PAGE')!=str(page_num): continue 
        for _, _, emcid in ks.inquire(subject=page_cid, property='EMBEDDING'):
            emkid, _ = ks.believe(page_cid, 'EMBEDDING', emcid)
            if get_property(emkid, 'MODEL') == embedding_model_name: 
                embeddings.append(np.array(json.loads(ds.recall(emcid))[0]))

    if not embeddings:
        return None

    return np.mean(np.stack(embeddings), axis=0)
    

predictions = []
print('Total documents to process:',len(document_cids))
for cid1 in document_cids:
    print('\n\n processing pdf: ', ds.encode(cid1))
    print(f"Progress: {(len(predictions)/len(document_cids)):.2%}\n\n")
    
    #d1em = get_page_embeddings(cid1, 1)
    d1em = get_page_embeddings(cid1)
    others = []
    for cid2 in document_cids:
        if cid1 == cid2: continue
        
        #d2em = get_page_embeddings(cid2, 1)
        d2em = get_page_embeddings(cid2)
        others.append((cid2,np.dot(d1em, d2em)))
    others = sorted(others, key=lambda x: x[1], reverse=True)
    predictions.append((cid1, get_property(cid1, 'assigned-class'), get_property(others[0][0], 'assigned-class'), others[0][1]))

correct = 0
for prediction in predictions:
    if prediction[1] == prediction[2]: correct+=1

print('accuracy:',correct/len(predictions))
