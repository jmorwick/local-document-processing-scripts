from io import BytesIO
from pdf2image import convert_from_bytes
import pytesseract
import cidnilib
from cidnilib import FileBasedDataService as FBDS
from cidnilib import PickleFileBasedDataService as PFBDS
from cidnilib import InMemoryKnowledgeService as IMKS
import PIL
#from paddleocr import PaddleOCR  # not currently working on python3.13
import numpy as np
import sys

if len(sys.argv) < 3:
   print("usage: python script pdf_list_cid ocr_model")

pdf_list_cid = sys.argv[1]
ocr_model_name = sys.argv[2]


ds = FBDS('./cidnidb')
kds = PFBDS('./cidnidb', levels=0)
ks = IMKS(kds)

PIL.Image.MAX_IMAGE_PIXELS = 500000000 

if ocr_model_name=='paddleocr':
    pocr = PaddleOCR(use_angle_cls=True, lang="en")

def paddleocr_process(pil_image):
    image_np = np.array(pil_image)
    result = pocr.ocr(image_np)
    lines = []
    for page in result:
        for line in page:
            text = line[1][0]
            lines.append(text)

    return "\n".join(lines)


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
            print("STARTING OCR FOR PAGE: ", pagenumber,'    Page CID:',page_cid)
            
            for _, _, ocr_cid in ks.inquire(subject=page_cid, property='OCR'):
                okid, _ = ks.believe(page_cid, 'OCR', ocr_cid)
                for _, _, cmodel in ks.inquire(subject=ds.encode(okid), property='MODEL'):
                    completed_models.add(cmodel)
                    print("** already computed: ",cmodel)
            if ocr_model_name in completed_models:
                print('already complete for',ocr_model_name,'with',page_cid)
                continue
            
            try:
                images = convert_from_bytes(ds.recall(page_cid), dpi=300)
                if len(images) != 1:
                    print('ERROR: PDF should have only one page for',page_cid,'from',ds.encode(cid),'...skipping')
                    continue
            except:
                print('ERROR: could not read page ' + page_cid)
                continue
            if ocr_model_name == 'tesseract':
                ocr_text = pytesseract.image_to_string(images[0])
            if ocr_model_name == 'paddleocr':
                paddleocr_process(images[0])
            else:
                print('ERROR: uncrecognized model name',ocr_model_name)
                sys.exit(1)
            print(ocr_text)
            ocid, _ = ds.know(ocr_text)
            okid, _ = ks.believe(page_cid, 'OCR', ds.encode(ocid))
            ks.believe(ds.encode(okid), 'MODEL', ocr_model_name)
            print('OCRTEXT: ',page_cid,ds.encode(ocid), '-->',ds.encode(okid))
            
kds.flush()   # move up to recover from interruptions more easily
