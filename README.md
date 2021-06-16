# ACSE 8: Machine Learning Module

This repository and README contain all the relevant information regarding module ACSE-8, please read it all carefully.

## Teaching structure:

The course is structured in 6 blocks containing the following topics:

1. Introduction to ML, supervised VS unsupervised ML, linear and logistic regression, and K-means and PCA.
2. Feed forward networks, backpropagation, and gradient descent
3. Regularisation, bias, and variance. Includes trainining, validation, and test sets.
4. Convolutional neural networks.
5. Recurrent neural networks and LSTMs. Intro to probability for ML.
6. Variational inference and generative models: VAEs and GANs (with a mention of normalising flows probably)

Every block will be covered in two consecutive days, the first one will cover the theory and the second will cover practical implementations. The day schedule for both theory and practical implementation days is:

| time  | session |
|---------------|-----------------------------------------------------------------------|
| 8:00h-9:00h   | early bird morning chat session to discuss any technical or other problems (optional) |
| 9:00h-12:00h  | lecture session |
| 13:00h-16:00h | exercise session |


**IMPORTANT NOTES ON CALENDAR DATES**:

No lectures are planned for: <br>
`Wednesday 5 May` <br>
and <br>
`Wednesday 12 May` <br>
However, there will be a morning lecture (exercises will be provided, but no afternoon session) on: <br>
`Wednesday 28 April` <br> 
to compensate for the fact that `Monday 3 May` is a UK bank holiday.

The practical implementation session 4 (CNNs) will be done on: <br>
`Thursday 6 May afternoon` <br>
and continued on <br>
`Friday 7 May afternoon` <br>
This is because coursework-1 (see below) will be done in the morning of `Friday 7 May` to ensure that students from all time zones can complete during daytime.


## Assessment

The course assessment will be based on two courseworks:

| dates | coursework description | weight |
|-----|---|---|
| Friday 7 May <br> 2 (hours)| **coursework-1**: time-limited written coursework (information will be added to this repo on Friday 7 morning). The coursework is scheduled to last two hours with an extra half an hour to submit it. | 50% |
| Tuesday 11 May (morning) until Sunday 16 May (23:59h) | **coursework-2**: coding-based exercise (information will be added to this repo on Tuesday 11 May) | 50% |

**MARKS**:

The final mark will be the equally weighted average between coursework-1 and coursework-2.

## Primer material and bibliography

###Introductory videos

To help prepare for the start of lectures we recommend you to watch these four short (15-20 mins) videos which provide a good introduction to Machine Learning:

1. [But what is a Neural Network? | Deep learning, chapter 1](https://www.youtube.com/watch?v=aircAruvnKk)  
2. [Gradient descent, how neural networks learn | Deep learning, chapter 2](https://www.youtube.com/watch?v=IHZwWFHWa-w&t=11s)   
3. [What is backpropagation really doing? | Deep learning, chapter 3](https://www.youtube.com/watch?v=Ilg3gGewQ5U)  
4. [Backpropagation calculus | Deep learning, chapter 4](https://www.youtube.com/watch?v=tIeHLnjs5U8)  

By the start of the third week we also recommend to watch these probability introductory videos to start getting familiar with some of the concepts introduced in blocks 5 and 6:

1. [Binomial distribution](https://www.youtube.com/watch?v=8idr1WZ1A7Q&list=RDCMUCYO_jab_esuFRV4b17AJtAw&index=2)
2. [Normal distribution](https://www.khanacademy.org/math/statistics-probability/modeling-distributions-of-data/more-on-normal-distributions/v/introduction-to-the-normal-distribution) 
3. [Bayes theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM) 


###Bibliography
- [Pattern recognition and Deep Learning (Christopher M. Bishop)](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf)
- [Deep Learning (Ian Goodfellow, Yoshua Bengio, Aaron Courville)](https://www.deeplearningbook.org/)


## Theory Lectures
Theory materials will be released on the morning of the lecture days, but often be they will be available ahead of this date (you can already access theory materials for blocks 1,2, and 3). <br>


| Session   |   Date/Time | Exercises |   Solutions   |
|-----------|-------------------|-----------------------|----------------------------|
| Introduction to Machine Learning [(slides)](https://github.com/acse-2020/ACSE-8/tree/main/theory/theory_1/morning_lecture/April26slides.pdf) | April 26 |  [Exercise 1](https://github.com/acse-2020/ACSE-8/tree/main/theory/theory_1/afternoon_exercises/Exercise_1/Exercise1Text.pdf) / [table](https://github.com/acse-2020/ACSE-8/tree/main/theory/theory_1/afternoon_exercises/Exercise_1/Table_for_presenting_results_without_the_solution.xlsx) <hr>  [Exercise 2](https://github.com/acse-2020/ACSE-8/tree/main/theory/theory_1/afternoon_exercises/Exercise_2/Exercise2Text.pdf)  |     [Solution 1](https://github.com/acse-2020/ACSE-8/tree/main/theory/theory_1/afternoon_exercises/Exercise_1/Exercise1ModelAnswer.pdf) / [solution table](https://github.com/acse-2020/ACSE-8/tree/main/theory/theory_1/afternoon_exercises/Exercise_1/Table_for_presenting_results_with_solution.xlsx) <hr> [Solution 2](https://github.com/acse-2020/ACSE-8/tree/main/theory/theory_1/afternoon_exercises/Exercise_2/Exercise2ModelAnswer.pdf) <br> [Houseprice dataset](https://github.com/acse-2020/ACSE-8/tree/main/theory/theory_1/afternoon_exercises/Exercise_2/Houseprices.csv) <br> [PCA regression](https://github.com/acse-2020/ACSE-8/tree/main/theory/theory_1/afternoon_exercises/Exercise_2/PCARegression.py)    |
| Feed-Foward Neural Networks [(slides)](https://github.com/acse-2020/ACSE-8/tree/main/theory/theory_2/morning_lecture/April28slides.pdf)  | April 28  |  [Exercise 1](https://github.com/acse-2020/ACSE-8/tree/main/theory/theory_2/afternoon_exercises/Exercise_1/Text_Exercise1.pdf) / [paper](https://github.com/acse-2020/ACSE-8/tree/main/theory/theory_2/afternoon_exercises/Exercise_1/SmilkovCarter.pdf) <hr> [Exercise 2](https://github.com/acse-2020/ACSE-8/tree/main/theory/theory_2/afternoon_exercises/Exercise_2/Text_Exercise2.pdf) / [code](https://github.com/acse-2020/ACSE-8/tree/main/theory/theory_2/afternoon_exercises/Exercise_2/HalfMoonsBackProp.py)|      [Solution 1](https://github.com/acse-2020/ACSE-8/tree/main/theory/theory_2/afternoon_exercises/Exercise_1/Model_Answer_Exercise1.pdf)  <hr>  [Solution 2](https://github.com/acse-2020/ACSE-8/tree/main/theory/theory_2/afternoon_exercises/Exercise_2/Model_answer_Exercise2.pdf)  |
| Regularization, Bias, Variance [(slides)](https://github.com/acse-2020/ACSE-8/tree/main/theory/theory_3/morning_lecture/April30slides.pdf)  | April 30  |  [Exercise 1](https://github.com/acse-2020/ACSE-8/tree/main/theory/theory_3/afternoon_exercises/Exercise_1/Text_Exercise_1.pdf) <hr> [Exercise 2](https://github.com/acse-2020/ACSE-8/tree/main/theory/theory_3/afternoon_exercises/Exercise_2/Text_Exercise_2.pdf) <hr>  [Exercise 3](theory/theory_3/afternoon_exercises/Exercise_3/Text_Exercise_3.pdf)  / [paper](https://github.com/acse-2020/ACSE-8/tree/main/theory/theory_3/afternoon_exercises/Exercise_3/Raschkapaper.pdf)|    [Solution 1](https://github.com/acse-2020/ACSE-8/tree/main/theory/theory_3/afternoon_exercises/Exercise_1/Model_answer_Exercise_1.pdf) <hr>  [Solution 2](https://github.com/acse-2020/ACSE-8/tree/main/theory/theory_3/afternoon_exercises/Exercise_2/Model_Answer_Exercise_2.pdf) <hr> [Solution 3](https://github.com/acse-2020/ACSE-8/tree/main/theory/theory_3/afternoon_exercises/Exercise_3/Model_answer_Exercise_3.pdf)   |
| Convolutional Neural Networks [(slides)](https://github.com/acse-2020/ACSE-8/blob/main/theory/theory_4/morning_lecture/May6slides.pdf)  | May 6  |    |        |
| Recurrent Networks and LSTMs / Probabilities for Deep Learning [(slides)](https://github.com/acse-2020/ACSE-8/blob/main/theory/theory_5/morning_lecture/RNNs_LSTM_and_intro_Probability.pdf)  | May 10  |  [Exercise 1](https://github.com/acse-2020/ACSE-8/blob/main/theory/theory_5/afternoon_exercises/Text%20Exercise%201.pdf)  <hr> [Exercise 2](https://github.com/acse-2020/ACSE-8/blob/main/theory/theory_5/afternoon_exercises/Text%20Exercise%202.pdf) |      [Solution 1](https://github.com/acse-2020/ACSE-8/blob/main/theory/theory_5/afternoon_exercises/Model%20answer%20Exercise%201.pdf) <hr> [Solution 2](https://github.com/acse-2020/ACSE-8/blob/main/theory/theory_5/afternoon_exercises/Model%20answer%20Exercise%202.pdf)  |
| Unsupervised Learning: Introduction to Generative Models: <br> [(slides VI)](https://github.com/acse-2020/ACSE-8/blob/main/theory/theory_6/morning_lecture/VariationalInference%20Autoencoders.pdf), [(Notes)](https://github.com/acse-2020/ACSE-8/blob/main/theory/theory_6/morning_lecture/VariationalInfernceNotes.pdf) <hr> [(slides GANs)](https://github.com/acse-2020/ACSE-8/blob/main/theory/theory_6/morning_lecture/GANs_theory_slides.pdf) | May 13  |  [Recommended reading](https://github.com/acse-2020/ACSE-8/blob/main/theory/theory_6/afternoon_exercises/AnIntroductionToDeepGenerativeModeling.pdf) |      <hr>     |

The links in the table will become active as we progress during the course.

## Implementation Lectures
Lecture slides/notebooks will be released on the morning before the start of the lectures. Two notebooks will be provided:

- **Exercise notebook**: Template with some code provided and empty blocks to work on the implementation during the lecture.
- **Solutions notebook**: Exercise notebook completed with model solutions for the tasks done during the lecture.

We encourage you to work on the **exercise notebook** and try to implement the tasks proposed during the lectures, but we acknowledge that some students will prefer to focus on the delivered material and understand the code well before attempting to implement it themselves.


## Implementation Lectures

| Session   |   Date/Time | Exercise (Colab) | Solutions (Colab)     |
|-----------|-------------------|-----------------------|----------------------------|
| Getting Started: Google Colab and Logistics | April 26 <br> 13:00-13:30  | [Colab intro](https://github.com/acse-2020/ACSE-8/blob/main/implementation/practical_0/Getting_Started.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-2020/ACSE-8/blob/main/implementation/practical_0/Getting_Started.ipynb) |    |
| Morning 1: Linear and Logistic Regression, k-Means, PCA | April 27 <br> 9:00-12:00  |  [Practical](https://github.com/acse-2020/ACSE-8/blob/main/implementation/practical_1/morning_lecture/Lecture1-Regression_KMeans_PCA_Practical.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-2020/ACSE-8/blob/main/implementation/practical_1/morning_lecture/Lecture1-Regression_KMeans_PCA_Practical.ipynb)   |     [Solutions](https://github.com/acse-2020/ACSE-8/blob/main/implementation/practical_1/morning_lecture/Lecture1-Regression_KMeans_PCA_Solutions.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-2020/ACSE-8/blob/main/implementation/practical_1/morning_lecture/Lecture1-Regression_KMeans_PCA_Solutions.ipynb) |
| Afternoon 1: Regression, K-Means, PCA continued | April 27 <br> 13:00-16:00  |  [Exercise](https://github.com/acse-2020/ACSE-8/blob/main/implementation/practical_1/afternoon_exercises/Afternoon1-Regression_KMeans_PCA_Practical.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-2020/ACSE-8/blob/main/implementation/practical_1/afternoon_exercises/Afternoon1-Regression_KMeans_PCA_Practical.ipynb) <br> [Additional Exercise](https://github.com/acse-2020/ACSE-8/blob/main/implementation/practical_1/afternoon_exercises/Afternoon_1_Datasets_Baselines_k_Means_PCA_Additional_Exercise.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-2020/ACSE-8/blob/main/implementation/practical_1/afternoon_exercises/Afternoon_1_Datasets_Baselines_k_Means_PCA_Additional_Exercise.ipynb) |   [Solutions](https://github.com/acse-2020/ACSE-8/blob/main/implementation/practical_1/afternoon_exercises/Afternoon1_Regression_KMeans_PCA_Solutions.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-2020/ACSE-8/blob/main/implementation/practical_1/afternoon_exercises/Afternoon1_Regression_KMeans_PCA_Solutions.ipynb)   <br>  [Additional Exercise Solutions](https://github.com/acse-2020/ACSE-8/blob/main/implementation/practical_1/afternoon_exercises/Afternoon_1_Datasets_Baselines_k_Means_PCA_Additional_Exercise_Solutions.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-2020/ACSE-8/blob/main/implementation/practical_1/afternoon_exercises/Afternoon_1_Datasets_Baselines_k_Means_PCA_Additional_Exercise_Solutions.ipynb)  |
| Morning 2: Pytorch, Automatic Differentiation, Neural Nets | April 29 <br> 9:00-12:00  |  [Practical](https://github.com/acse-2020/ACSE-8/blob/main/implementation/practical_2/morning_lecture/Morning_Session_2_Pytorch_Autograd_Optimization_Neural_Networks_Practical.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-2020/ACSE-8/blob/main/implementation/practical_2/morning_lecture/Morning_Session_2_Pytorch_Autograd_Optimization_Neural_Networks_Practical.ipynb)   |     [Solutions](https://github.com/acse-2020/ACSE-8/blob/main/implementation/practical_2/morning_lecture/Morning_Session_2_Pytorch_Autograd_Optimization_Neural_Networks_Practical_Solutions.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-2020/ACSE-8/blob/main/implementation/practical_2/morning_lecture/Morning_Session_2_Pytorch_Autograd_Optimization_Neural_Networks_Practical_Solutions.ipynb)      |
| Afternoon 2: Optimization and improving Neural Nets <br> [PyBryt version of Exercise 1](https://classroom.github.com/a/tQizuCSE) <br> & [Video instructions](https://web.microsoftstream.com/video/019c025f-8d39-4645-bf94-afde9f5b332d?list=studio) | April 29 <br> 13:00-16:00  |  [Exercise_1](https://classroom.github.com/a/tQizuCSE) <br> [Exercises 2_3](https://github.com/acse-2020/ACSE-8/blob/main/implementation/practical_2/afternoon_exercises/Afternoon_Session_2_Diving_into_Optimization_and_Neural_Nets_Exercises_2_and_3.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-2020/ACSE-8/blob/main/implementation/practical_2/afternoon_exercises/Afternoon_Session_2_Diving_into_Optimization_and_Neural_Nets_Exercises_2_and_3.ipynb) |     [Solutions Exercise 1](https://github.com/acse-2020/ACSE-8/blob/main/implementation/practical_2/afternoon_exercises/Afternoon_Session_2_Diving_into_Optimization_and_Neural_Nets_Exercise_1_Solution.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-2020/ACSE-8/blob/main/implementation/practical_2/afternoon_exercises/Afternoon_Session_2_Diving_into_Optimization_and_Neural_Nets_Exercise_1_Solution.ipynb) <br> [Solutions Exercises 2_3](https://github.com/acse-2020/ACSE-8/blob/main/implementation/practical_2/afternoon_exercises/Afternoon_Session_2_Diving_into_Optimization_and_Neural_Nets_Exercises_2_and_3_Solutions.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-2020/ACSE-8/blob/main/implementation/practical_2/afternoon_exercises/Afternoon_Session_2_Diving_into_Optimization_and_Neural_Nets_Exercises_2_and_3_Solutions.ipynb)   |
| Morning 3: MNIST, Neural Networks,Regularization, Cross-Validation | May 4 <br> 9:00-12:00  |  [Practical](https://github.com/acse-2020/ACSE-8/blob/main/implementation/practical_3/morning_lecture/Morning_Session_3_MNIST_Neural_Networks_Regularization_Cross_Validation.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-2020/ACSE-8/blob/main/implementation/practical_3/morning_lecture/Morning_Session_3_MNIST_Neural_Networks_Regularization_Cross_Validation.ipynb)   |     [Solutions](https://github.com/acse-2020/ACSE-8/blob/main/implementation/practical_3/morning_lecture/Morning_Session_3_MNIST_Neural_Networks_Regularization_Cross_Validation_Solutions.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-2020/ACSE-8/blob/main/implementation/practical_3/morning_lecture/Morning_Session_3_MNIST_Neural_Networks_Regularization_Cross_Validation_Solutions.ipynb)  |
| Afternoon 3: Grid Search | May 4 <br> 13:00-16:00 |  [Exercise](https://github.com/acse-2020/ACSE-8/blob/main/implementation/practical_3/afternoon_exercises/Afternoon_Session_3_GridSearch_Exercise.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-2020/ACSE-8/blob/main/implementation/practical_3/afternoon_exercises/Afternoon_Session_3_GridSearch_Exercise.ipynb) |     [Solutions](https://github.com/acse-2020/ACSE-8/blob/main/implementation/practical_3/afternoon_exercises/Afternoon_Session_3_GridSearch_Exercise_Solutions.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-2020/ACSE-8/blob/main/implementation/practical_3/afternoon_exercises/Afternoon_Session_3_GridSearch_Exercise_Solutions.ipynb)      |
| From Convolution to ConvNet - part 1 | May 6 <br> 13:00-16:00  |  [Practical](https://github.com/acse-2020/ACSE-8/blob/main/implementation/practical_4/afternoon_lecture_1/Afternoon_Session_4_part1_FromConvolutions_To_ConvNets_Practical.ipynb) <br>  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-2020/ACSE-8/blob/main/implementation/practical_4/afternoon_lecture_1/Afternoon_Session_4_part1_FromConvolutions_To_ConvNets_Practical.ipynb)  |   [Solutions](https://github.com/acse-2020/ACSE-8/blob/main/implementation/practical_4/afternoon_lecture_1/Afternoon_Session_4_part1_FromConvolutions_To_ConvNets_Practical_Solutions.ipynb)   <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-2020/ACSE-8/blob/main/implementation/practical_4/afternoon_lecture_1/Afternoon_Session_4_part1_FromConvolutions_To_ConvNets_Practical_Solutions.ipynb) |
| From Convolution to ConvNet - part 2 | May 7 <br> 13:00-16:00  |  [Practical](https://github.com/acse-2020/ACSE-8/blob/main/implementation/practical_4/afternoon_lecture_2/Morning_Session_4_Transfer_Learning_Exercise.ipynb) <br>  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-2020/ACSE-8/blob/main/implementation/practical_4/afternoon_lecture_2/Morning_Session_4_Transfer_Learning_Exercise.ipynb)  |    [Solutions](https://github.com/acse-2020/ACSE-8/blob/main/implementation/practical_4/afternoon_lecture_2/Morning_Session_4_Transfer_Learning_Solutions.ipynb) <br>  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-2020/ACSE-8/blob/main/implementation/practical_4/afternoon_lecture_2/Morning_Session_4_Transfer_Learning_Solutions.ipynb)    |
| Morning 5: Recurrent Neural Networks | May 11 <br> 9:00-12:00  |  [Practical](https://github.com/acse-2020/ACSE-8/blob/main/implementation/practical_5/morning_lecture/Practical5_RecurrentNeuralNetworks_Exercises.ipynb) <br>  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-2020/ACSE-8/blob/main/implementation/practical_5/morning_lecture/Practical5_RecurrentNeuralNetworks_Exercises.ipynb) |  [Solutions](https://github.com/acse-2020/ACSE-8/blob/main/implementation/practical_5/morning_lecture/Practical5_RecurrentNeuralNetworks_Solutions.ipynb) <br>  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-2020/ACSE-8/blob/main/implementation/practical_5/morning_lecture/Practical5_RecurrentNeuralNetworks_Solutions.ipynb) |
| Afternoon 5: Exercise Recurrent Neural Networks | May 11 <br> 13:00-16:00 |  [use morning notebook]  |  [use morning notebook]      | 
| Morning 6: Generative Models | May 14 <br> 9:00-12:00  | [Practical VAEs](https://github.com/acse-2020/ACSE-8/blob/main/implementation/practical_6/morning_lecture/Practical6_VariationalAutoEncoder_blanks.ipynb) <br>  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-2020/ACSE-8/blob/main/implementation/practical_6/morning_lecture/Practical6_VariationalAutoEncoder_blanks.ipynb) <br>  [Practical GANs](https://github.com/acse-2020/ACSE-8/blob/main/implementation/practical_6/morning_lecture/Morning_Session_6_GANs_Exercise_vanillaGAN.ipynb) <br>  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-2020/ACSE-8/blob/main/implementation/practical_6/morning_lecture/Morning_Session_6_GANs_Exercise_vanillaGAN.ipynb)   <br>   [Practical cGANs](https://github.com/acse-2020/ACSE-8/blob/main/implementation/practical_6/morning_lecture/Morning_Session_6_GANs_Exercise_conditionalGAN.ipynb) <br>  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-2020/ACSE-8/blob/main/implementation/practical_6/morning_lecture/Morning_Session_6_GANs_Exercise_conditionalGAN.ipynb) |  [Solutions VAEs](https://github.com/acse-2020/ACSE-8/blob/main/implementation/practical_6/morning_lecture/Practical6_VariationalAutoEncoder.ipynb) <br>  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-2020/ACSE-8/blob/main/implementation/practical_6/morning_lecture/Practical6_VariationalAutoEncoder.ipynb)  <br>  [Solutions GANs](https://github.com/acse-2020/ACSE-8/blob/main/implementation/practical_6/morning_lecture/Morning_Session_6_GANs_Solution_vanillaGAN.ipynb) <br>  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-2020/ACSE-8/blob/main/implementation/practical_6/morning_lecture/Morning_Session_6_GANs_Solution_vanillaGAN.ipynb) <br>   [Solutions cGANs](https://github.com/acse-2020/ACSE-8/blob/main/implementation/practical_6/morning_lecture/Morning_Session_6_GANs_Solution_conditionalGAN.ipynb) <br>  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-2020/ACSE-8/blob/main/implementation/practical_6/morning_lecture/Morning_Session_6_GANs_Solution_conditionalGAN.ipynb) | 
| Afternoon 6: wrap up | May 14 <br> 13:00-16:00  |  [Final session](https://github.com/acse-2020/ACSE-8/blob/main/implementation/practical_6/afternoon_final_session/Final_session.ipynb) <br>  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/acse-2020/ACSE-8/blob/main/implementation/practical_6/afternoon_final_session/Final_session.ipynb)  |  [Olivier's talk](https://github.com/acse-2020/ACSE-8/blob/main/implementation/practical_6/afternoon_final_session/PresentationGANs_Olivier.pdf)  |




## Remote teaching

Microsoft Teams is being used for all remote teaching. We recommend that you install the Teams app both on your personal computer/laptop and smartphone. You have all been added to the ACSE20 Team. You are expected to check for module updates on Teams in addition to email. If it is not possible for you to use Teams for whatever reason then you must contact the course director, Dr Gerard Gorman, immediately.

You should have received calendar invites for all teaching sessions – you can always go into the Calendar view in Teams to join the session. When you connect to the lecture:

- Please do not click the Record button.
- You should mute your microphone and keep it muted throughout the entire lecture.
- You should post any comments/questions using the Teams chat where the teaching assistants (TA’s) will assist you.
- During each lecture one of the teaching team will also act as moderator and interject when appropriate so that the lecturer can address popular questions on the chat.
- All academic questions should be posted on the class channel so that the whole class can benefit from seeing the question and answer.
- Give thumbs up to questions posted by other people that you also want to ask, so that the moderator can easily identify them and interject to make sure they are addressed by the lecturer.

While all lectures will be recorded, you are expected to attend the online lectures and exercise sessions. These sessions are the only times that TA’s have been booked to support you online. Support outside these times will be limited as the staff and grad students involved also have other commitments.

It is your responsibility to check your Imperial College Outlook/Teams calendar for the exact schedule and any updates, ensuring that you have taken into account your local time zone. Because we have students spread across the globe we are unable to shift the timetable without disadvantaging students either east or west of BST.


## Teaching Team
### Module coordinators

- Lluis Guasch [(email)](mailto:lguasch@imperial.ac.uk) 
- Olivier Dubrule [(email)](mailto:o.dubrule@imperial.ac.uk)

### GTAs
- Oscar Bates
- Carlos Cueto
- Yao Jiashun
- Deborah Pelacani Cruz
- George Strong
- Zainab D Titus



All lecture recordings will be available here *(link will be live as soon as we start the course)*

## Module objectives:

Have fun!

Over the next three weeks, we will go from here:

![](https://imgs.xkcd.com/comics/machine_learning.png) <br>
[XKCD 1838](https://xkcd.com/1838/)

to understanding complex network architectures, how and why they work, and when to use them:

| ![](https://static.cambridge.org/binary/version/id/urn:cambridge.org:id:binary:20201223160655562-0200:S2053470120000311:S2053470120000311_fig10.png) |
|------|
[*deep variational autoencoders*](https://www.cambridge.org/core/journals/design-science/article/generation-of-geometric-interpolations-of-building-types-with-deep-variational-autoencoders/F4899EC122329816CD137503D8118875)
