
## Binary Classification

This is my Machine Learning project at unsw, for the berryjam competition, the challenge is the following:

Problem: Building image classifiers requires vast amounts of data, powerful computing resources, and complex engineering. If we could build highly accurate models with minimal data, we make image AI accessible for a lot more domains and organisations.

Challenge: Develop an algorithm and an automated process to train a domain-specific image classifier using only 5 positive and 5 negative images. This methodology should be adaptable across different narrowly scoped problems simply by changing the provided training data. For example:

Identifying ripe fruit

Detecting faulty circuit boards

Spotting cracks in pipelines

Distinguishing between weeds and crops

We are approaching the problem using transfer learning, by finetuning 3 pretrained models for the specific tasks then using ensemble learning techniques to choose the correct prediction.

Follow instructions the to generate the automated process of the image classifier.

# Instructions

Steps to run:

1- Make a virtual environment with python version 3.10.5
2- Run 1_install.sh
3- Run 2_run_pipelines.sh with appropriate directories


Main Function:
--------------

1. main(train_input_dir: str, train_labels_file_name: str, target_column_name: str, train_output_dir: str):
   The main function responsible for training the model.

   Arguments:
   - train_input_dir (str): Path to the folder with the CSV and training images.
   - train_labels_file_name (str): Name of the CSV file containing labels.
   - target_column_name (str): Name of the target column within the CSV file.
   - train_output_dir (str): Path to the folder to save training output.


Usage:
------

Example usage:

python predict.py -d "path/to/Data - Is Epic Intro Full" -l "Is Epic Files.txt" -t "Is Epic" -m "path/to/Is Epic/model" -o "path/to/Is Epic Full Predictions.csv"

Reference:

- Huggingface.com
- How to finetune a pretrained image classification model

