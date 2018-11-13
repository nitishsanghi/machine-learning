# Machine Learning Engineer Nanodegree
## Capstone Proposal
Nitish Sanghi

November  8th, 2018

## The Proposal
### Domain Background
Transportation technology have exponentially evolved over the last century, both in speed, comfort, and safety. The evolution has democratized transportation empowering people. The generation of transportation technology is taking advantage
of the tremendous break throughs made in Robotics, Computing, and Hardware tech. Truly "Autonomous Vehicles" whether on land, air or underwater have now become a reality opening up transportation solutions which not only will make transportation faster, safe, and efficient but also has the potential to make a dent in our fight to stop/slow climate change. Autnomous Vehicles require massive amounts of real world data to be captured, processed, and interpreted to make autonomous decisions to manuveur the vehicle to its destination in a safe manner. It is an extremely hard problem requiring integration of sensors with the vehicle which capture data to be processed by the computers on board the vehicle, tracking features like road signs, lanes, vehicles, pedestrians, traffic lights, and a number of other things that we might take for garantted when we drive.

Various computer vision techniques have been used in the past to identify and characterize features an autonomous vehicle might encounter as it moves. One of the most fundamental classifications that a vehicle will have to make is classifying traffic signs and other vehicles which are moving around it. The methodologies in deep learning have provided powerful tools which when integrated with basic computer vision techniques can "teach" a vehicle to identify features discussed above and classify them appropriately. I am interested in building a classifier to identify different road signs and distinguish them from cars moving along side the vehicle and other features like trees, road curbs, road dividers, etc.

ADD ACADEMIC REFERENCES 

### Problem Statement
The problem that I am proposing to solve is to design a deep learning network to classify road signs, vehicles, and other features as misc which are captured by cameras on an autonomous vehicle.

ADD Metrics

### Datasets and Inputs
The dataset being used is comprised of road sign images, vehicle images captured by cameras placed on the front right, left, and centre of the vehicle, and miscellaneous features like trees, sky, road, curbs, dividers etc. These are some of the features that an autonomous vehicle would encounter as it navigates from point A to B. Efficient and safe functioning of the autonomous vehicle requires classification and identification of features in appropriate catergories so as to help the decision making process of the autonomous vehicle. 

The datasets were obtained from other Udacity courses.The road sign images have 43 different classifications and have been shot at differt angles, distances, and time of day which provides enough variation for training a network. The vehicle and miscellaneous datasets are split into vehicle and non-vehicles categories. Some of the vehicles capture the back end of the vehicles and some the sides and the back end. The vehicles images are also captured from different angles (left bias or right bias). 

The datasets will be input into a Convolutional Neural Network (CNN) to train it classify the road signs appropriately, vehicles as vehicles, and miscellaneous features. Before the data is fed into the CNN it will be processed and augmented to increase training data size. Using these datasets a CNN can be trained to identify some basic/simple features an autonomous vehicle will encounter while navigating. A potential extension to this can be further classification of the features independently like type of surrounding vehicle, type of miscellaneous features, etc in a complete image frame captured by camera sensors.

### Solution Statement
The solution for the proposed problem is output of a trained CNN which classifies test inputs to appropriates classes. The classes are 43 road signs, vehicle, and non-vehicle. In all there are 45 classes for the CNN to classify the inputs to.

### Benchmark Model
LeNet 5 will be used as the benchmark model. The indented solution will be classification of the the various road signs, vehicles, and non-vehicles in the correct categories. The results of the benchmark model will be compared using the "Accuracy" as the evaluation metric which is the percentage of correct classifications.

### Evaluation Metrics
To evaluate the efficiency of the network "Accuracy" will be used as the metric. For example, if a 1000 images are input to the CNN and the network correctly classifies 937 images, then the accuracy would be 93.7%. The evaluation metric is relatively simple and straight forward. Both benchmark and solution model will be evaluated using the "Accuracy" as the evaluation metric.

### Project Design


In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
