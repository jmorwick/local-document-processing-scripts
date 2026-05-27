from io import BytesIO
import cidnilib
import numpy as np
from cidnilib import FileBasedDataService as FBDS
from cidnilib import PickleFileBasedDataService as PFBDS
from cidnilib import InMemoryKnowledgeService as IMKS
from openai import OpenAI
import PIL
import json
import sys


if len(sys.argv) < 3:
   print("usage: python script pdf_list_cid prompt_template_cid ocr_model_name llm_model_name data-matrix-file (1nn/2nn/1-1nn)", file=sys.stderr)

pdf_list_cid = sys.argv[1]
prompt_template_cid = sys.argv[2]
ocr_model_name = sys.argv[3]
llm_model_name = sys.argv[4]
data_file_name = sys.argv[5]
strategy = sys.argv[6]
first_page_only = True
processor_name = llm_model_name + '-' + strategy+','+prompt_template_cid
matrix = np.loadtxt(data_file_name, delimiter=",")

if strategy not in {'1nn', '1-1nn', '2nn'}:
    print('ERROR: unknown strategy',strategy, file=sys.stderr)
    sys.exit(1)

print(f"doing {strategy} on {data_file_name} with shape {matrix.shape} using llm {llm_model_name} over {ocr_model_name} data using template {prompt_template_cid}", file=sys.stderr)

ds = FBDS('./cidnidb')
kds = PFBDS('./cidnidb', levels=0)
ks = IMKS(kds)


document_cids = []

for line in ds.recall(ds.decode(pdf_list_cid)).decode().split('\n'):
        line = line.strip() 
        if not line: continue
        cid = ds.decode(line)   
        document_cids.append(cid)
        


# query class list
class_list = set()
for doc_cid in document_cids:
    for _, _, class_label in ks.inquire(ds.encode(doc_cid), 'assigned-class'):
        class_list.add(class_label) 
        
print('using class list: ',class_list)
        
        
def get_property(cid, property):
    if type(cid) == bytes: cid = ds.encode(cid) 
    try: return next(ks.inquire(cid, property))[2]
    except: return None
        

llm_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

def classify_with_llm(prompt):
    return llm_client.chat.completions.create(
        model=llm_model_name,
        messages=[
            {"role": "user", "content": prompt_text}
        ]
    ).choices[0].message.content

print('Total documents to process:',len(document_cids), file=sys.stderr)
r = 0
for doc_cid in document_cids:
    print('\n\n processing pdf: ', ds.encode(doc_cid))
    print(f"Progress: {(r/len(document_cids)):.2%}\n\n")
            
    
    # find rag content
    row = matrix[r]
    others = []
    for c in range(len(row)):
        if doc_cid == document_cids[c]: continue
        others.append((document_cids[c],row[c]))
    others = sorted(others, key=lambda x: x[1], reverse=True)
    most_similar_cid = None
    most_similar_similarity = None
    most_similar_class = None
    second_most_similar_cid = None
    second_most_similar_similarity = None
    second_most_similar_class = None
    for other_cid, other_similarity in others:
        if not most_similar_cid:
            most_similar_cid = other_cid
            most_similar_similarity = other_similarity
            most_similar_class = get_property(other_cid, 'assigned-class')
            if strategy == '1nn': break
            continue
        other_class = get_property(other_cid, 'assigned-class')
        if strategy == '1-1nn' and other_class == most_similar_class: continue
        # doing 2-nn...
        second_most_similar_cid = other_cid
        second_most_similar_similarity = other_similarity
        second_most_similar_class = other_class
        break
    r += 1
    
    
    
    # check if already done
    completed_models = set()
    for _, _, class_label in ks.inquire(subject=ds.encode(doc_cid), property='CLASSIFICATION'):
        okid, _ = ks.believe(ds.encode(doc_cid), 'CLASSIFICATION', class_label)
        for _, _, new_processor_name in ks.inquire(subject=ds.encode(okid), property='MODEL_AND_PROMPT'):
            completed_models.add(new_processor_name)
            print(f"** already computed: {new_processor_name}->{class_label}")
    if processor_name in completed_models:
        print('already complete for',processor_name)
        #kds.forget(okid)
        continue
    
    
    # gather up all pages to be used for classification
    page_text = dict()
    for _, prop, page_cid in ks.inquire(subject=ds.encode(doc_cid)):
        if prop != 'CONTAINS': continue
        kid, _ = ks.believe(ds.encode(doc_cid), prop, page_cid)
        _, _, pagenumber = list(ks.inquire(subject=ds.encode(kid), property='PAGE'))[0]
        if first_page_only and pagenumber != '1': 
            continue
        for _, _, ocr_cid in ks.inquire(subject=page_cid, property='OCR'):
            okid, _ = ks.believe(page_cid, 'OCR', ocr_cid)
            for _, _, cmodel in ks.inquire(subject=ds.encode(okid), property='MODEL'):
                if ocr_model_name == cmodel:
                    page_text[int(pagenumber)] = ds.recall_text(ocr_cid)   
    full_text = ''
    for page_num in sorted(page_text):
        full_text += page_text[page_num]
    page_text = dict()
    for _, prop, page_cid in ks.inquire(subject=ds.encode(most_similar_cid)):
        if prop != 'CONTAINS': continue
        kid, _ = ks.believe(ds.encode(most_similar_cid), prop, page_cid)
        _, _, pagenumber = list(ks.inquire(subject=ds.encode(kid), property='PAGE'))[0]
        if first_page_only and pagenumber != '1': 
            continue
        for _, _, ocr_cid in ks.inquire(subject=page_cid, property='OCR'):
            okid, _ = ks.believe(page_cid, 'OCR', ocr_cid)
            for _, _, cmodel in ks.inquire(subject=ds.encode(okid), property='MODEL'):
                if ocr_model_name == cmodel:
                    page_text[int(pagenumber)] = ds.recall_text(ocr_cid)   
    most_similar_text = ''
    for page_num in sorted(page_text):
        most_similar_text += page_text[page_num]
    page_text = dict()
    if second_most_similar_cid: 
        for _, prop, page_cid in ks.inquire(subject=ds.encode(second_most_similar_cid)):
            if prop != 'CONTAINS': continue
            kid, _ = ks.believe(ds.encode(second_most_similar_cid), prop, page_cid)
            _, _, pagenumber = list(ks.inquire(subject=ds.encode(kid), property='PAGE'))[0]
            if first_page_only and pagenumber != '1': 
                continue
            for _, _, ocr_cid in ks.inquire(subject=page_cid, property='OCR'):
                okid, _ = ks.believe(page_cid, 'OCR', ocr_cid)
                for _, _, cmodel in ks.inquire(subject=ds.encode(okid), property='MODEL'):
                    if ocr_model_name == cmodel:
                        page_text[int(pagenumber)] = ds.recall_text(ocr_cid)   
    second_most_similar_text = ''
    for page_num in sorted(page_text):
        second_most_similar_text += page_text[page_num]
        
        
    # form prompt and execute
    try: 
        prompt_text = f"{ds.recall_text(prompt_template_cid)}\n{'\n'.join(class_list)}\n"
        prompt_text += f"MOST SIMILAR CLASS: {most_similar_class}\nSIMILARITY: {most_similar_similarity}\nTEXT: {most_similar_text}\n"
        if strategy in ('1-1nn', '2nn'):
            prompt_text += f"SECOND MOST SIMILAR CLASS: {second_most_similar_class}\nSIMILARITY: {second_most_similar_similarity}\nTEXT: {second_most_similar_text}\n"
        prompt_text += f"\nOCR BEGIN\n\n{full_text}"
        print('--------------',prompt_text,'\n------------------')
        response = classify_with_llm(prompt_text)
        print(response,'\n-------------')
        class_label, confidence = response.strip().split('\n')[-1].split(',')
        class_label = class_label.strip()
        confidence = confidence.strip()
        print(f"'{class_label}', '{confidence}'")
    except Exception as e:
        print('ERROR: LLM did not return a prediction',e)
        class_label = 'None'
        confidence = '0%'

    okid, _ = ks.believe(ds.encode(doc_cid), 'CLASSIFICATION', class_label)
    okid, _ = ks.believe(ds.encode(okid), 'MODEL_AND_PROMPT', processor_name)
    ks.believe(ds.encode(okid), 'CONFIDENCE', confidence)
    print('CLASSIFICATION: ',ds.encode(doc_cid), '-->',class_label, confidence)
    kds.flush()   # move up to recover from interruptions more easily

