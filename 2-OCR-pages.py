from io import BytesIO
from pdf2image import convert_from_bytes
import pytesseract
import cidnilib
from cidnilib import FileBasedDataService as FBDS
from cidnilib import PickleFileBasedDataService as PFBDS
from cidnilib import InMemoryKnowledgeService as IMKS
import PIL

ds = FBDS('./cidnidb')
kds = PFBDS('./cidnidb', levels=0)
ks = IMKS(kds)
modelname = 'tesseract'
PIL.Image.MAX_IMAGE_PIXELS = 500000000

skips=['GuJCq5RSRH15jXpSGZdPYZ4uvMXpvNJG6Rsd13ZXpaPDwr']

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
            print("STARTING OCR FOR PAGE: ", pagenumber,'    Page CID:',page_cid)
            
            for _, _, ocr_cid in ks.inquire(subject=page_cid, property='OCR'):
                okid, _ = ks.believe(page_cid, 'OCR', ocr_cid)
                for _, _, cmodel in ks.inquire(subject=ds.encode(okid), property='MODEL'):
                    completed_models.add(cmodel)
                    print("** already computed: ",cmodel)
            if modelname in completed_models:
                print('already complete for',modelname,'with',page_cid)
                continue
            
            try:
                images = convert_from_bytes(ds.recall(page_cid), dpi=300)
                if len(images) != 1:
                    print('ERROR: PDF should have only one page for',page_cid,'from',ds.encode(cid),'...skipping')
                    continue
            except:
                print('ERROR: could not read page ' + page_cid)
                continue
            ocr_text = pytesseract.image_to_string(images[0])
            ocid, _ = ds.know(ocr_text)
            okid, _ = ks.believe(page_cid, 'OCR', ds.encode(ocid))
            ks.believe(ds.encode(okid), 'MODEL', modelname)
            print('OCRTEXT: ',page_cid,ds.encode(ocid), '-->',ds.encode(okid))
            kds.flush()
