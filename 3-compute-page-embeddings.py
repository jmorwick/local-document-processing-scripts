from io import BytesIO
import cidnilib
from sentence_transformers import SentenceTransformer
import numpy as np
from cidnilib import FileBasedDataService as FBDS
from cidnilib import PickleFileBasedDataService as PFBDS
from cidnilib import InMemoryKnowledgeService as IMKS
import PIL
import json

ds = FBDS('./cidnidb')
kds = PFBDS('./cidnidb', levels=0)
ks = IMKS(kds)
ocrmodel = 'tesseract'
modelname = 'BAAI/bge-small-en-v1.5'
PIL.Image.MAX_IMAGE_PIXELS = 500000000

skips=['GuJCq5RSRH15jXpSGZdPYZ4uvMXpvNJG6Rsd13ZXpaPDwr']



embedding_model = SentenceTransformer(modelname)   

def embed_text(text: str) -> np.ndarray:
    return np.asarray(
            embedding_model.encode(
                list(text),
                normalize_embeddings=True
            )
        )[0]


with open("original-pdf-ids.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()  
        cid = ds.decode(line)
        print('\n\n NEW PDF: ', line,'\n\n')
        
        for _, prop, page_cid in ks.inquire(subject=ds.encode(cid)):
            if page_cid in skips: continue
            if prop != 'CONTAINS': continue
            kid, _ = ks.believe(ds.encode(cid), prop, page_cid)
            completed_models = set()
            _, _, pagenumber = list(ks.inquire(subject=ds.encode(kid), property='PAGE'))[0]
            print("STARTING Embedding computation FOR PAGE: ", pagenumber,'    Page CID:',page_cid)
            
            for _, _, ocr_cid in ks.inquire(subject=page_cid, property='EMBEDDING'):
                okid, _ = ks.believe(page_cid, 'EMBEDDING', ocr_cid)
                for _, _, cmodel in ks.inquire(subject=ds.encode(okid), property='MODEL'):
                    completed_models.add(cmodel)
                    print("** already computed: ",cmodel)
            if modelname in completed_models:
                print('already complete for',modelname,'with',page_cid)
                continue
            
            ocrtext = None
            for _, _, ocr_cid in ks.inquire(subject=page_cid, property='OCR'):
                okid, _ = ks.believe(page_cid, 'OCR', ocr_cid)
                for _, _, cmodel in ks.inquire(subject=ds.encode(okid), property='MODEL'):
                    if ocrmodel == cmodel:
                        ocrtext = ds.recall_text(ocr_cid)
            
            if ocrtext:
                    embeddings = embed_text(ocrtext)
                    print("Embedded text: ", embeddings.shape)
                    embeddings_as_text = json.dumps(embeddings.tolist()) # restore with np.array(json.loads(as_text))
                    emcid, _ = ds.know(embeddings_as_text)
                    okid, _ = ks.believe(page_cid, 'EMBEDDING', ds.encode(emcid))
                    ks.believe(ds.encode(okid), 'MODEL', modelname)
                    print('EMBEDDING: ',page_cid,ds.encode(emcid), '-->',ds.encode(okid))
