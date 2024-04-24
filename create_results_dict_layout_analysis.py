''' 
Script to create a results dictionary containing features values and information about correct/incorrect segmentation
input: predicted regions and corresponding ground truth, separated folders with correctly and incorrectly segmented region pairs
output: results dictionary in json format
'''

import glob
import os
import pandas as pd
import numpy as np
import json
from bs4 import BeautifulSoup as bs
from tqdm import tqdm
from PIL import Image
from shapely.geometry import box
from scipy.spatial.distance import directed_hausdorff
import Levenshtein
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'path/to/tesseract.exe'
model_text_presence = 'GT4HistOCR'
model_ocr = 'frak2021'

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# function to get intersection over union
def get_iou(box1, box2):
    intersection = box1.intersection(box2).area
    iou = intersection/(box1.area + box2.area - intersection)
    return iou

# function to get coordinates from the xml file (specific for the xml page format)
def get_predicted_coordinates(region):
    x = int(region['hpos'])
    y = int(region['vpos'])
    width = int(region['width'])
    height = int(region['height'])

    predicted_coordinates = (x, y, (x+width), (y+height))
    return predicted_coordinates

# function to derive texts similarity from Levenshtein distance
def count_text_similarity(distance, text1, text2):
    if max(len(text1), len(text2)) > 0:
        similarity = 1 - (distance / max(len(text1), len(text2)))
    else:
        similarity = None
    return similarity

# formate coordinates as needed
def rectangle_to_points(coordinates):
    x1, y1, x2, y2 = coordinates
    points = np.array([(x1, y1), (x1, y2), (x2, y1), (x2, y2)])
    return points

# tags of the newspapers we want to evaluate
tags = ['apr', 'awi', 'aze', 'btb', 'dvb', 'fdb', 'fst', 'gre', 'gtb', 'ibn', 
        'krz', 'lvb', 'mop', 'nfp', 'nwb', 'nwg', 'nwj', 'pab', 'pel', 'pit', 
        'ptb', 'rpt', 'sch', 'svb', 'tpt', 'vlz', 'vtl', 'vvb', 'wrz']

# loading our annotations
json_files_paths = glob.glob('annotations/*')

# concat all annotations in one dataframe
df = pd.DataFrame()
for path in json_files_paths:
    file_df = pd.read_json(path_or_buf=path, lines=True)
    df = pd.concat([df, file_df], axis=0)

# drop duplicite files, reset index afterwards
df = df.drop_duplicates(subset="filename")
df = df.reset_index()
df = df.drop(columns=["index"])

# create a column containing information whether the given page contains ads or not
df["contains_ads"] = df["bbox"].apply(lambda x: 1 if x else 0)

# create a new ads dataframe only from pages containing ads
ads = df.loc[df.contains_ads == 1]
ads = ads.reset_index()

# add column with tag and coordinates for every given page
ads['tag'] = [ads['filename'][i][0:3] for i in range(len(ads))]
ads["coordinates"] = ads["bbox"].apply(lambda bbox_list: [[bbox["x"], bbox["y"], bbox["height"], bbox["width"]] for bbox in bbox_list])

results_dict = {}

# go through all publications tags and create a new dataframe for every publication
for tag in tags:
    print(tag)

    tag_df = ads[ads['tag']==tag]
    tag_df = tag_df.reset_index()
    pages = list(tag_df.filename)

    tag_results = {}

    # go through all pages in the tag dataframe
    for page in tqdm(pages):
        xml_path = f'anno_xml_pages/{tag}/{page[:-4]}.xml' 
        img_path = f'samples_images/{tag}/{page[:-4]}.jpg'  
        img = Image.open(img_path)
        page_results = {}

        # if the prediction exists, get all regions from the xml file for every given page, and save their coordinates
        page_predicted_regions = []

        if os.path.exists(xml_path):
            with open(xml_path, "r", encoding='utf-8') as file:
                content = file.readlines()
                content = "".join(content)
                bs_content = bs(content, "lxml")
            
            regions = bs_content.find_all('textblock')
            for region in regions:
                predicted_coordinates = get_predicted_coordinates(region)
                page_predicted_regions.append(predicted_coordinates) # here are saved the predicted regions

            page_annotated_coors = list(ads.loc[ads.filename == page].coordinates) # and here we have the annotated regions

            for coor in page_annotated_coors[0]:           # for every annotated job ads region:
                box1 = box(coor[0], coor[1], (coor[0] + coor[3]), (coor[1] + coor[2]))
                highest_intersection = 0
                
                for k in range(len(page_predicted_regions)):   # for every coordinate from anno:
                    box2 = box(page_predicted_regions[k][0], page_predicted_regions[k][1], page_predicted_regions[k][2], page_predicted_regions[k][3]) ### predicted region
                    if box1.intersection(box2).area > highest_intersection:
                        highest_intersection = box1.intersection(box2).area
                        corresponding_region_index = k

                if highest_intersection > 0:
                    k = corresponding_region_index

                    box1 = box(coor[0], coor[1], (coor[0] + coor[3]), (coor[1] + coor[2])) ### annotated region
                    box2 = box(page_predicted_regions[k][0], page_predicted_regions[k][1], page_predicted_regions[k][2], page_predicted_regions[k][3]) ### predicted region

                    # calculate text similarity
                    cropped1 = img.crop((coor[0], coor[1], (coor[0] + coor[3]), (coor[1] + coor[2])))
                    cropped2 = img.crop((page_predicted_regions[k][0], page_predicted_regions[k][1], page_predicted_regions[k][2], page_predicted_regions[k][3]))

                    text1 = pytesseract.image_to_string(cropped1, lang=model_ocr)
                    text2 = pytesseract.image_to_string(cropped2, lang=model_ocr)

                    distance = Levenshtein.distance(text1, text2)
                    similarity = count_text_similarity(distance, text1, text2)

                    # calculate intersection
                    intersection = box1.intersection(box2).area
                    relative_intersection = intersection/(max(box1.area, box2.area))
                    iou = get_iou(box1, box2)
                   
                    box1_coordinates = (int(coor[0]), int(coor[1]), int(coor[0] + coor[3]), int(coor[1] + coor[2]))
                    box2_coordinates = (int(page_predicted_regions[k][0]), int(page_predicted_regions[k][1]), int(page_predicted_regions[k][2]), int(page_predicted_regions[k][3]))

                    points_box1 = rectangle_to_points(box1_coordinates)
                    points_box2 = rectangle_to_points(box2_coordinates)
                    hausdorff_distance = directed_hausdorff(points_box1, points_box2)[0]
                    
                    # identify text presence in non-intersecting areas:
                    borders_contain_text = False
                    
                    # lower edge: 
                    if box1_coordinates[3] != box2_coordinates[3]:
                        lower_edge = img.crop((min(box1_coordinates[0], box2_coordinates[0]), min(box1_coordinates[3], box2_coordinates[3]), max(box1_coordinates[2], box2_coordinates[2]), max(box1_coordinates[3], box2_coordinates[3])))
                        ocr_result = pytesseract.image_to_string(lower_edge, lang=model_text_presence)
                        if any(char.isalpha() for char in ocr_result):
                            borders_contain_text=True

                    # upper edge: 
                    if box1_coordinates[1] != box2_coordinates[1]:
                        upper_edge = img.crop((min(box1_coordinates[0], box2_coordinates[0]), min(box1_coordinates[1], box2_coordinates[1]), max(box1_coordinates[2], box2_coordinates[2]), max(box1_coordinates[1], box2_coordinates[1])))
                        ocr_result = pytesseract.image_to_string(upper_edge, lang=model_text_presence)
                        if any(char.isalpha() for char in ocr_result):
                            borders_contain_text=True

                    # right edge: 
                    if box1_coordinates[2] != box2_coordinates[2]:
                        right_edge = img.crop((min(box1_coordinates[2], box2_coordinates[2]), min(box1_coordinates[1], box2_coordinates[1]), max(box1_coordinates[2], box2_coordinates[2]), max(box1_coordinates[3], box2_coordinates[3])))
                        ocr_result = pytesseract.image_to_string(right_edge, lang=model_text_presence)
                        if any(char.isalpha() for char in ocr_result):
                            borders_contain_text=True

                    # left edge: 
                    if box1_coordinates[0] != box2_coordinates[0]:
                        left_edge = img.crop((min(box1_coordinates[0], box2_coordinates[0]), min(box1_coordinates[1], box2_coordinates[1]), max(box1_coordinates[0], box2_coordinates[0]), max(box1_coordinates[3], box2_coordinates[3])))
                        ocr_result = pytesseract.image_to_string(left_edge, lang=model_text_presence)
                        if any(char.isalpha() for char in ocr_result):
                            borders_contain_text=True

                    path_to_correctly_segmented = f'pairs_evaluation/correct/{page}_{k}_annotated.tif'
                    path_to_incorrectly_segmented = f'pairs_evaluation/incorrect/{page}_{k}_annotated.tif'

                    if os.path.exists(path_to_correctly_segmented):
                        page_results[str(k)] = {'Intersection': relative_intersection, 'IoU': iou, 'Levenshtein': similarity, 'BordersText': borders_contain_text, 'Hausdorff': hausdorff_distance, 'CorrectSegmentation': True}
                    elif os.path.exists(path_to_incorrectly_segmented):
                        page_results[str(k)] = {'Intersection': relative_intersection, 'IoU': iou, 'Levenshtein': similarity, 'BordersText': borders_contain_text, 'Hausdorff': hausdorff_distance, 'CorrectSegmentation': False}
                    else:
                        page_results[str(k)] = {'Intersection': relative_intersection, 'IoU': iou, 'Levenshtein': similarity, 'BordersText': borders_contain_text, 'Hausdorff': hausdorff_distance, 'CorrectSegmentation': None}

            tag_results[page] = page_results

        # for given tag, save results to an overall results dictionary
        results_dict[tag] = tag_results

output_path = 'features_dict.json'
with open(output_path, 'w') as json_file:
    json.dump(results_dict, json_file) 
