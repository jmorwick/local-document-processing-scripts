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

print(f"reading direct classifications for {pdf_list_cid} from llm {llm_model_name} with prompt {llm_model_name}", file=sys.stderr)

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
        


predictions = []
print('Total documents to process:',len(document_cids), file=sys.stderr)
r = 0
for doc_cid in document_cids:
    print('\n\n processing pdf: ', ds.encode(doc_cid), file=sys.stderr)
    print(f"Progress: {(len(predictions)/len(document_cids)):.2%}\n\n", file=sys.stderr)
       
    

            
    class_label = 'None'
    confidence = '0%'        
    for _, _, class_label in ks.inquire(subject=ds.encode(doc_cid), property='CLASSIFICATION'):
        okid, _ = ks.believe(ds.encode(doc_cid), 'CLASSIFICATION', class_label)
        for okid2, _, other_processor_name in ks.inquire(subject=ds.encode(okid), property='MODEL_AND_PROMPT'):
            if processor_name == other_processor_name:
                try: 
                    stored_confidence = get_property(okid2, 'CONFIDENCE').strip()
                    print(stored_confidence)
                    if stored_confidence[-1] == '%': stored_confidence = stored_confidence[:-1]
                    confidence = float(stored_confidence)/100.0 if int(stored_confidence)>1 else float(stored_confidence) 
                except:
                    print('ERROR: corrupt prediction')

    predictions.append((ds.encode(doc_cid), get_property(doc_cid, 'assigned-class'), class_label, confidence))
    print(','.join(map(str,predictions[-1])))
    r += 1
correct = 0
for prediction in predictions:
    if prediction[1] == prediction[2]: correct+=1

print('accuracy:',correct/len(predictions))

