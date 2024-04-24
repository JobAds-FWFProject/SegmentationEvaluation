'''This code compares the ability of models to identify text in the images. 
Assumed files structure:

images contatining text:
border_samples/contain_text/img1.jpg
                            /img2.jpg
                            /img3.jpg ...

images not containing any text:
border_samples/no_text/img1.jpg
                        /img2.jpg
                        /img3.jpg ...

Outuput: Count of True positives, false negatives, true negatives and false positives for the given model.'''

import glob
from PIL import Image
from tqdm import tqdm
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'path/to/Tesseract-OCR/tesseract.exe'

def identify_text_presence(img_path, model):
    img = Image.open(img_path)
    ocr_result = pytesseract.image_to_string(img, lang=model)
    return 1 if any(char.isalpha() for char in ocr_result) else 0

def count_text_images(images_folder):
    count = 0
    for path in tqdm(images_folder):
        count += identify_text_presence(path, model)
    return count

model = 'frk' # 'deu' 'frk' 'deu_frak' 'GT4HistOCR' 'frak2021'

text_folder = 'border_samples/contain_text/*'
no_text_folder = 'border_samples/no_text/*'

text_paths = glob.glob(text_folder)
no_text_paths = glob.glob(no_text_folder)

text_identified_in_text_images = count_text_images(text_paths)
text_identified_in_images_without_text = count_text_images(no_text_paths)

print(f'The {model} model identified text in {text_identified_in_text_images} out of {len(text_paths)} images. \
      This means {text_identified_in_text_images} True Positives and {len(text_paths)-text_identified_in_text_images} False Negatives.')
print(f'The {model} model identified text in {text_identified_in_images_without_text} out of {len(no_text_paths)} images. \
      This means {len(no_text_paths)-text_identified_in_images_without_text} True Negatives and {text_identified_in_images_without_text} False Positives.')
