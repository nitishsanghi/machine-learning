# Machine Learning Engineer Nanodegree
## Capstone Proposal
Nitish Sanghi

November  8th, 2018

## The Proposal
### Domain Background
In recent years, the credit card issuers in Taiwan faced the cash and credit card debt crisis and the delinquency is expected to peak in the third quarter of 2006 (Chou, 2006). In order to increase market share, card-issuing banks in Taiwan over-issued cash and credit cards to unqualified applicants. At the same time, most cardholders, irrespective of their repayment ability, overused credit card for consumption and accumulated heavy credit and cash– card debts. The crisis caused the blow to consumer finance confidence and it is a big challenge for both banks and cardholders.
In a well-developed financial system, crisis management is on the downstream and risk prediction is on the upstream. The major purpose of risk prediction is to use financial information, such as business financial statement, customer transaction and repayment records, etc., to predict business performance or individual customers’ credit risk and to reduce the damage and uncertainty.
Many statistical methods, including discriminant analy- sis, logistic regression, Bayes classifier, and nearest neigh- bor, have been used to develop models of risk prediction (Hand & Henley, 1997). With the evolution of artificial intelligence and machine learning, artificial neural net- works and classification trees were also employed to fore- cast credit risk (Koh & Chan, 2002; Thomas, 2000). Credit risk here means the probability of a delay in the repayment of the credit granted (Paolo, 2001).
From the perspective of risk control, estimating the probability of default will be more meaningful than classi- fying customers into the binary results – risky and non- risky. Therefore, whether or not the estimated probability of default produced from data mining methods can repre- sent the ‘‘real” probability of default is an important prob- lem. To forecast probability of default is a challenge facing practitioners and researchers, and it needs more study (Baesens, Setiono, Mues, & Vanthienen, 2003; Baesens et al., 2003; Desai, Crook, & Overstreet, 1996; Hand &
  
Because the real probability of default is unknown, this study proposed the novel ‘‘Sorting Smoothing Method” to deduce the real default probability and offered the solu- tions to the following two questions:
(1) Is there any difference of classification accuracy among the six data mining techniques?
(2) Could the estimated probability of default produced from data mining methods represent the real proba- bility of default?
In the next section, we review the six data mining tech- niques (discriminant analysis, logistic regression, Bayes classifier, nearest neighbor, artificial neural networks, and classification trees) and their applications on credit scoring. Then, using the real cardholders’ credit risk data in Tai- wan, we compare the classification accuracy among them. Section 4 is dedicated to the predictive performance of probability of default among them. Finally, Section 5 con- tains some concluding remarks.

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

The first step is to collect the appropriate datasets and split them into training, validation, and test sets. The next step is to explore the datasets. Check for all the classes present in the datasets. The studying the distribution of different classes which can be easily represented as an histogram is important. The dataset classes need to have enough data points to train the network for high accuracy. Next step would be to process the data. The data can be augmented by applying image filters like mirror, swirl, gaussian blurring, perspective changes, etc. Once the data has been sufficiently processed the next step is to train the CNN and iterate while changing parts of the CNN to optimize the "Loss" of the network. 

The next part is to implement LeNet 5 which is the benchmark model. Both networks will be trained and validated using the same datasets. Once the networks are trained, the test dataset is fed into both networks. The networks are going to output the predicted classes of the test data. Based on the number of corrected classifications, the accuracy of the two networks will be worked out. The goal is the design a CNN which performs as good as or better than the benchmark model. 


-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
