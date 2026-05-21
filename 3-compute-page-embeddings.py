from io import BytesIO
import cidnilib
from sentence_transformers import SentenceTransformer
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


print('computing embeddings with',embedding_model_name,'using OCR from',ocr_model_name)

embedding_model = SentenceTransformer(embedding_model_name)   



ds = FBDS('./cidnidb')
kds = PFBDS('./cidnidb', levels=0)
ks = IMKS(kds)

def embed_text(text: str) -> np.ndarray:
    return embedding_model.encode(
                [text],
                normalize_embeddings=True
            )


document_cids = ds.recall(ds.decode(pdf_list_cid)).decode().split('\n')
print('Total documents to process:',len(document_cids))
processed = 0
for line in ds.recall(ds.decode(pdf_list_cid)).decode().split('\n'):
        processed += 1
        line = line.strip() 
        if not line: continue
        cid = ds.decode(line)   
        
        print('\n\n processing pdf: ', line)
        print(f"Progress: {(processed/len(document_cids)):.2%}\n\n")
        
        for _, prop, page_cid in ks.inquire(subject=ds.encode(cid)):
            if prop != 'CONTAINS': continue
            kid, _ = ks.believe(ds.encode(cid), prop, page_cid)
            completed_models = set()
            _, _, pagenumber = list(ks.inquire(subject=ds.encode(kid), property='PAGE'))[0]
            print("STARTING Embedding computation FOR PAGE: ", pagenumber,'    Page CID:',page_cid)
            
            for _, _, emcid in ks.inquire(subject=page_cid, property='EMBEDDING'):
                okid, _ = ks.believe(page_cid, 'EMBEDDING', emcid)
                for _, _, cmodel in ks.inquire(subject=ds.encode(okid), property='MODEL'):
                    completed_models.add(cmodel)
                    print("** already computed: ",cmodel)

            if embedding_model_name in completed_models:
                print('already complete for',embedding_model_name,'with',page_cid)
                continue
            
            ocrtext = None
            for _, _, ocr_cid in ks.inquire(subject=page_cid, property='OCR'):
                okid, _ = ks.believe(page_cid, 'OCR', ocr_cid)
                for _, _, cmodel in ks.inquire(subject=ds.encode(okid), property='MODEL'):
                    if ocr_model_name == cmodel:
                        ocrtext = ds.recall_text(ocr_cid)
            
            if ocrtext:
                    embeddings = embed_text(ocrtext)
                    print("Embedded text: ", embeddings.shape)
                    embeddings_as_text = json.dumps(embeddings.tolist())
                    emcid, _ = ds.know(embeddings_as_text)
                    okid, _ = ks.believe(page_cid, 'EMBEDDING', ds.encode(emcid))
                    ks.believe(ds.encode(okid), 'MODEL', embedding_model_name)
                    print('EMBEDDING: ',page_cid,ds.encode(emcid), '-->',ds.encode(okid))
                    
kds.flush()   # move up to recover from interruptions more easily
