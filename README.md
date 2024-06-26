
## Binary Classification

This is my Machine Learning project at unsw, for the berryjam competition, the challenge is the following:

Problem: Building image classifiers requires vast amounts of data, powerful computing resources, and complex engineering. If we could build highly accurate models with minimal data, we make image AI accessible for a lot more domains and organisations.

Challenge: Develop an algorithm and an automated process to train a domain-specific image classifier using only 5 positive and 5 negative images. This methodology should be adaptable across different narrowly scoped problems simply by changing the provided training data. For example:

Identifying ripe fruit

Detecting faulty circuit boards

Spotting cracks in pipelines

Distinguishing between weeds and crops

We are approaching the problem using transfer learning, by finetuning 3 pretrained models for the specific tasks then using ensemble learning techniques to choose the correct prediction.

Note: Follow instructions in readme.txt to generate the automated process of the image classifier.
