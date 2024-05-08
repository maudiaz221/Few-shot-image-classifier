import argparse
import os
from typing import Any

import pandas as pd
from PIL import Image
import torch

from common import load_model, load_predict_image_names, load_single_image
from transformers import AutoModelForImageClassification, AutoImageProcessor


########################################################################################################################

def parse_args():
    """
    Helper function to parse command line arguments
    :return: args object
    """
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--predict_data_image_dir', required=True, help='Path to image data directory')
    parser.add_argument('-l', '--predict_image_list', required=True,
                        help='Path to text file listing file names within predict_data_image_dir')
    parser.add_argument('-t', '--target_column_name', required=True,
                        help='Name of column to write prediction when generating output CSV')
    parser.add_argument('-m', '--trained_model_dir', required=True,
                        help='Path to directory containing the model to use to generate predictions')
    parser.add_argument('-o', '--predicts_output_csv', required=True, help='Path to CSV where to write the predictions')
    args = parser.parse_args()
    return args


#Prediction based on answers of the 3 models first 2 models perform better than the last one
#That is why weights are assigned as 2,2,1
def predict(models: Any, image: Image, target_column_name:str) -> str:
    
    yes = 0
    no = 0
    path_to_models = f'resources/pretrained/models/{target_column_name}'
    weights = [2, 1.5, 0.5]  # Weights for the three models, adjust as needed
    for model, weight in zip(models, weights): #We are also using the weights to give more importance to some models
        image_processor = AutoImageProcessor.from_pretrained(os.path.join(path_to_models, model))
        model = AutoModelForImageClassification.from_pretrained(os.path.join(path_to_models, model))
        #we load the model and the processor from the finetuned models and give an output for the prediction
        encoding = image_processor(image.convert("RGB"), return_tensors="pt")
        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits
        
        predicted_class_idx = logits.argmax(-1).item()
        if predicted_class_idx == 0:
            yes += weight
            
        else:
            no += weight
        
    
    if yes >= no:
        return 'Yes'
    else:
        return 'No'
    



def main(predict_data_image_dir: str,
         predict_image_list: str,
         target_column_name: str,
         trained_model_dir: str,
         predicts_output_csv: str):
    """
    The main body of the predict.py responsible for:
     1. load model
     2. load predict image list
     3. for each entry,
           load image
           predict using model
     4. write results to CSV

    :param predict_data_image_dir: The directory containing the prediction images.
    :param predict_image_list: Name of text file within predict_data_image_dir that has the names of image files.
    :param target_column_name: The name of the prediction column that we will generate.
    :param trained_model_dir: Path to the directory containing the model to use for predictions.
    :param predicts_output_csv: Path to the CSV file that will contain all predictions.
    """

    # load pre-trained models or resources at this stage.
    models = load_model(trained_model_dir, target_column_name)

    # Load in the image list
    image_list_file = os.path.join(predict_data_image_dir, predict_image_list)
    image_filenames = load_predict_image_names(image_list_file)

    # Iterate through the image list to generate predictions
    predictions = []
    for filename in image_filenames:
        try:
            image_path = os.path.join(predict_data_image_dir, filename)
            image = load_single_image(image_path)
            label = predict(models, image,target_column_name)
            predictions.append(label)
        except Exception as ex:
            print(f"Error generating prediction for {filename} due to {ex}")
            predictions.append("Error")

    df_predictions = pd.DataFrame({'Filenames': image_filenames, target_column_name: predictions})

    # Finally, write out the predictions to CSV
    df_predictions.to_csv(predicts_output_csv, index=False)


if __name__ == '__main__':
    """
    Example usage:

    python predict.py -d "path/to/Data - Is Epic Intro Full" -l "Is Epic Files.txt" -t "Is Epic" -m "path/to/Is Epic/model" -o "path/to/Is Epic Full Predictions.csv"

    """
    args = parse_args()
    predict_data_image_dir = args.predict_data_image_dir
    predict_image_list = args.predict_image_list
    target_column_name = args.target_column_name
    trained_model_dir = args.trained_model_dir
    predicts_output_csv = args.predicts_output_csv

    main(predict_data_image_dir, predict_image_list, target_column_name, trained_model_dir, predicts_output_csv)

########################################################################################################################
