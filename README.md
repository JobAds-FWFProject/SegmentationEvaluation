# SegmentationEvaluation

The repository accompanies the paper 'Quantifying Page Segmentation Quality for Historical Job Advertisements Retrieval' (Venglarova, Adam, Balasubramanian, Vogeler). For all references and further explanation, please refer to the paper.

## text_identification_comparison.py
This code serves to compare the ability of different models to identify text in manually labeled images.

## create_results_dict_layout_analysis.py
This code produces a json dictionary of features values for individual images and information about correct/incorrect segmentation. The images being compared are the predicted region and its ground truth.

## features_evaluation.py
This code serves to fit and evaluate logistic regression models based on the dictionary produced in the create_results_dict_layout_analysis.py script.

## logit_model_Hausdorff_BordersText.pkl
The resulting trained model that takes Hausdorff distance and Text Presence information and classifies a segment as correctly or incorrectly classified.
