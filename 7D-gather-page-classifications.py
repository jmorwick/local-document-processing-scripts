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
   print("usage: python script pdf_list_cid prompt_template_cid ocr_model_name llm_model_name")

pdf_list_cid = sys.argv[1]
prompt_template_cid = sys.argv[2]
ocr_model_name = sys.argv[3]
llm_model_name = sys.argv[4]


processor_name = llm_model_name+','+prompt_template_cid

print(f"finding predictions from {processor_name} on {pdf_list_cid}", file=sys.stderr)

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
  
def get_pages(cid):
    embeddings = []
    page_cids = dict()
    for _, _, page_cid in ks.inquire(ds.encode(cid), 'CONTAINS'):
        kid, _ = ks.believe(ds.encode(cid), 'CONTAINS', page_cid)
        page_cids[get_property(kid, 'PAGE')] = page_cid
    return page_cids


def get_page_embeddings(cid):
    for _, _, emcid in ks.inquire(subject=cid, property='EMBEDDING'):
        emkid, _ = ks.believe(cid, 'EMBEDDING', emcid)
        if get_property(emkid, 'MODEL') == embedding_model_name: 
            return np.array(json.loads(ds.recall(emcid))[0])
    
         
print('Total documents to process:',len(document_cids), file=sys.stderr)
r = 0
all_pages = []
first_pages = set()
for cid1 in document_cids: 
    print('\n\n loading pdf pages: ', ds.encode(cid1), file=sys.stderr)
    print(f"Progress: {(r/len(document_cids)):.2%}\n\n", file=sys.stderr)

    pages = get_pages(cid1)
    if '1' not in pages: 
        print(f"ERROR: no first page (all pages: {pages}", file=sys.stderr)
        continue
    all_pages.extend([pages[pnum] for pnum in sorted(pages)])
    r += 1
    
r = 0   


predictions = []
print('Total pages to process:',len(document_cids), file=sys.stderr)
r = 0
for cid in all_pages:
    print('\n\n processing page: ', cid, file=sys.stderr)
    print(f"Progress: {(len(predictions)/len(all_pages)):.2%}\n\n", file=sys.stderr)
    class_label = 'None'
    confidence = '0'        
    for _, _, new_class_label in ks.inquire(cid, property='CLASSIFICATION'):
        okid, _ = ks.believe(cid, 'CLASSIFICATION', new_class_label)
        for _, _, other_processor_name in ks.inquire(subject=ds.encode(okid), property='MODEL_AND_PROMPT'):
            if processor_name == other_processor_name:
                class_label = new_class_label
                try: 
                    okid2, _ = ks.believe(ds.encode(okid), 'MODEL_AND_PROMPT',processor_name)
                    stored_confidence = get_property(okid2, 'CONFIDENCE')
                    stored_confidence = stored_confidence.strip()
                    if stored_confidence[-1] == '%': stored_confidence = stored_confidence[:-1]
                    confidence = float(stored_confidence)/100.0 if int(stored_confidence)>1 else float(stored_confidence) 
                except:
                    okid2, _ = ks.believe(ds.encode(okid), 'MODEL_AND_PROMPT',processor_name)
                    print(f"ERROR: corrupt prediction: '{class_label}', '{get_property(okid2, 'CONFIDENCE')}'", file=sys.stderr)

    predictions.append((cid, cid in first_pages, class_label=='FIRST', confidence))
    print(predictions[-1][0]+','+str(predictions[-1][1])+','+str(predictions[-1][2])+','+str(float(predictions[-1][3])))
    r += 1
correct = 0
for prediction in predictions:
    if prediction[1] == prediction[2]: correct+=1

print('accuracy:',correct/len(predictions), file=sys.stderr)

