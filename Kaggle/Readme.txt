

Steps to run:

1- Make a virtual environment with python version 3.10.5
2- Run 1_install.sh
3- Run 2_run_pipelines.sh with appropriate directories

----------------------------------------------------------------------------------------------------------------------------------------------------
train.py Documentation
========================


1. load_train_resources(resource_dir: str = 'resources') -> Any:
   Load any resources (i.e., pre-trained models, data files, etc.) from the specified directory.

   Arguments:
   - resource_dir (str, optional): The relative directory from `train.py` where resources are kept. Default is 'resources'.

   Returns:
   - Any: TBD (To Be Determined)

2. pre_process(image_processor: AutoImageProcessor) -> Tuple[Callable, Callable]:
   Pre-processes the images by applying transformations for data augmentation.

   Arguments:
   - image_processor (AutoImageProcessor): Pre-trained image processor for normalization and resizing.

   Returns:
   - Tuple[Callable, Callable]: Two functions for pre-processing training and validation images, respectively.

3. aquire_labels(dataset) -> Tuple[Dict[str, int], Dict[int, str]]:
   Acquires label mappings from the dataset.

   Arguments:
   - dataset: The dataset object containing train and test splits.

   Returns:
   - Tuple[Dict[str, int], Dict[int, str]]: Mapping dictionaries for label-to-id and id-to-label.

4. collate_fn(examples) -> Dict[str, Tensor]:
   Collates examples into batches for training.

   Arguments:
   - examples: List of examples containing image pixels and labels.

   Returns:
   - Dict[str, Tensor]: A dictionary containing pixel values and labels.

Model Training Functions:
--------------------------

1. compute_metrics(eval_pred) -> float:
   Computes the F1 score metric on a batch of predictions.

   Arguments:
   - eval_pred: The evaluation prediction object containing predictions and label ids.

   Returns:
   - float: The computed F1 score.

2. train(dataset, models, output_directory: str) -> List[str]:
   Trains a classification model using the training dataset and specified pre-trained models.

   Arguments:
   - dataset: The dataset object containing train and test splits.
   - models: List of pre-trained model paths.
   - output_directory (str): The directory to save trained models.

   Returns:
   - List[str]: List of paths to the fine-tuned model directories.

Main Function:
--------------

1. main(train_input_dir: str, train_labels_file_name: str, target_column_name: str, train_output_dir: str):
   The main function responsible for training the model.

   Arguments:
   - train_input_dir (str): Path to the folder with the CSV and training images.
   - train_labels_file_name (str): Name of the CSV file containing labels.
   - target_column_name (str): Name of the target column within the CSV file.
   - train_output_dir (str): Path to the folder to save training output.





common.py Documentation
========================

1. load_image_labels(labels_file_path: str) -> pd.DataFrame:
   Loads the labels from a CSV file.

   Arguments:
   - labels_file_path (str): Path to the CSV file containing the image labels.

   Returns:
   - pd.DataFrame: A pandas DataFrame containing the loaded labels.

2. makeFolder(df: pd.DataFrame, folder_path: str, target_column_name: str) -> str:
   Makes a folder with the appropriate separation of images based on the labels.

   Arguments:
   - df (pd.DataFrame): DataFrame containing image filenames and labels.
   - folder_path (str): Path to the folder containing the images.
   - target_column_name (str): The name of the target column.

   Returns:
   - str: Path to the created folder.

3. load_predict_image_names(predict_image_list_file: str) -> [str]:
   Reads a text file with one image filename per line and returns a list of filenames.

   Arguments:
   - predict_image_list_file (str): Path to the text file containing the image filenames.

   Returns:
   - list: A list of image filenames.

4. load_single_image(image_file_path: str) -> Image:
   Load an image from file.

   Arguments:
   - image_file_path (str): Path to the image file.

   Returns:
   - Image: The loaded image.

Model Loading and Saving Functions:
------------------------------------

1. save_model(model: Any, target: str, output_dir: str):
   Save a model to disk.

   Arguments:
   - model (Any): The model object to save.
   - target (str): The target value - can be useful to name the model file for the target it is intended for.
   - output_dir (str): The output directory to save the model file(s).

   Note:
   - This function is not implemented. You need to implement it according to your specific requirements.

2. load_model(trained_model_dir: str, target_column_name: str) -> Any:
   Load a model from disk.

   Arguments:
   - trained_model_dir (str): The directory where the model file(s) are saved.
   - target_column_name (str): The target value - can be useful to name the model file for the target it is intended for.

   Returns:
   - Any: The loaded model(s).

   Note:
   - This function returns a list of model filenames in the given directory. You may need to load the actual models based on these filenames.



predict.py Documentation
========================

This script is responsible for predicting labels for images using a weighted ensemble of pre-trained models.

Functions:
----------

1. parse_args():
   Helper function to parse command line arguments.

   Returns:
   - args: An argparse.Namespace object containing parsed arguments.

2. predict(models: Any, image: Image, target_column_name: str) -> str:

   Predicts a label for the given image using a weighted ensemble of models.

   Arguments:
   - models (Any): A list of models to use for prediction.
   - image (Image): The image to make predictions on.
   - target_column_name (str): The name of the target column.

   Returns:
   - str: The predicted label.

3. main(predict_data_image_dir: str, predict_image_list: str, target_column_name: str, trained_model_dir: str, predicts_output_csv: str):

   The main body of the predict.py responsible for:
   1. Load model.
   2. Load predict image list.
   3. For each entry, load image and predict using model.
   4. Write results to CSV.

   Arguments:
   - predict_data_image_dir (str): The directory containing the prediction images.
   - predict_image_list (str): Name of text file within predict_data_image_dir that has the names of image files.
   - target_column_name (str): The name of the prediction column that we will generate.
   - trained_model_dir (str): Path to the directory containing the model to use for predictions.
   - predicts_output_csv (str): Path to the CSV file that will contain all predictions.

Usage:
------

Example usage:

python predict.py -d "path/to/Data - Is Epic Intro Full" -l "Is Epic Files.txt" -t "Is Epic" -m "path/to/Is Epic/model" -o "path/to/Is Epic Full Predictions.csv"

Reference:

- Huggingface.com
- How to finetune a pretrained image classification model
