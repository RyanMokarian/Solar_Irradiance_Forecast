'''
*****************************************************************************
*	Title: "Solar Irradiance Prediction"

*	Category: Machine Learning

*	Authors: Ryan Mokarian, Olivier Tessier-Lariviere, Azfar Khoja, Yichen Lin
*****************************************************************************
	Description:
The goal of solar irradiance prediction is to help with the
electric grid management. This is a need that is increasingly important as solar
energy continue to expand. In this project, various deep learning
approaches (Convolutional Neural Network 2D and 3D, LSTM, GRU and Attention) compared
for the task of solar irradiance nowcasting at a speciﬁc location using
satellite imagery. The best RMSE by implicitly using the Clearsky model
to predict GHI was obtained. For details, refer to the uploaded Manuscript.
'''


## Run the Project


Title: Solar Irradiance Prediction 
"IFT6759 course: Advanced Projects in Machine Learning"

Category: Web Application, Data Base

Author: Colabortation of Ryan Mokarian, Mani Ahmadi and Bilal Qandeel

Description: /* In this project, a database for a university created and hosted in the Concordia University server (faculty MySQL DBMS) and a php based web application designed to connect and interact with the DB server. */


## How to Visualize Training with Tensorboard
Open an ssh session :
```
ssh -L 16006:127.0.0.1:6006 guest@helios3.calculquebec.ca
```
Run tensorboard : 
```
tensorboard --logdir=solar-irradiance/logs
```
Open the following url in your browser : 
```
http://127.0.0.1:16006
```
