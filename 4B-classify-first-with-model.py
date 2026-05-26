from io import BytesIO
from openai import OpenAI
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

def get_property(cid, property):
    if type(cid) == bytes: cid = ds.encode(cid) 
    try: return next(ks.inquire(cid, property))[2]
    except: return None

ds = FBDS('./cidnidb')
kds = PFBDS('./cidnidb', levels=0)
ks = IMKS(kds)



document_cids = []
for line in ds.recall(ds.decode(pdf_list_cid)).decode().split('\n'):
        line = line.strip() 
        if not line: continue
        cid = ds.decode(line)   
        document_cids.append(cid)
   
print('using template',prompt_template_cid,'with model',llm_model_name,'to classify documents as first page or not')

prompt_template = ds.recall(prompt_template_cid).decode()

llm_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

def classify_with_llm(ocr_text):
    return llm_client.chat.completions.create(
        model=llm_model_name,
        messages=[
            {"role": "user", "content": f"{prompt_template}\nOCR BEGIN\n\n{ocr_text}"}
        ]
    ).choices[0].message.content

    


predictions = 0
print('Total documents to process:',len(document_cids))
for doc_cid in document_cids:
        print('\n\n processing pdf: ', ds.encode(doc_cid))
        print(f"Progress: {(predictions/len(document_cids)):.2%}\n\n")
        predictions += 1
       
        
        # gather up all pages to be used for classification
        page_text = dict()
        for _, prop, page_cid in ks.inquire(subject=ds.encode(doc_cid)):
            if prop != 'CONTAINS': continue
            kid, _ = ks.believe(ds.encode(doc_cid), prop, page_cid)
            _, _, pagenumber = list(ks.inquire(subject=ds.encode(kid), property='PAGE'))[0]
            
            for _, _, ocr_cid in ks.inquire(subject=page_cid, property='OCR'):
                okid, _ = ks.believe(page_cid, 'OCR', ocr_cid)
                for _, _, cmodel in ks.inquire(subject=ds.encode(okid), property='MODEL'):
                    if ocr_model_name == cmodel:
                        
                        cfcid,_ = ks.believe(ocr_cid, 'CLASSIFICATION_FIRST_PAGE', 'None')
                        kds.forget(cfcid)
                        print('page',pagenumber)
                        try: 
                            response = classify_with_llm(ds.recall_text(ocr_cid))
                            print(response)
                            class_label, confidence = response.strip().split('\n')[-1].split(',')
                            class_label = class_label.strip()
                            confidence = confidence.strip()
                            print(f"page {pagenumber}: '{class_label}', '{confidence}'")
                        except:
                            print('ERROR: LLM did not return a prediction')
                            class_label = 'None'
                            confidence = '0%'
                        okid, _ = ks.believe(ocr_cid, 'CLASSIFICATION_FIRST_PAGE', class_label)
                        okid, _ = ks.believe(ds.encode(okid), 'MODEL_AND_PROMPT', processor_name)
                        ks.believe(ds.encode(okid), 'CONFIDENCE', confidence)
                        print('CLASSIFICATION_FIRST_PAGE: ',ocr_cid, '-->',class_label, confidence)
            kds.flush()   # move up to recover from interruptions more easily
