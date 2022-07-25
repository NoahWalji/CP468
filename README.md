# CP468: Heart Disease Predictor
CP48 Final Course Assignment
The concern for heart disease is growing, and with that the goal of this project is to determine accurate predictors of an individuals chance of being diagnosed with heart disease.
The dataset being used consists of a variety of different health conditions and habits an individual has such as alochol consumption, BMI and kidney disease.
The dataset is a public dataset on the website Kaggle and can be found in this repo or through the following link: 

https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease

The project places the data into two seperate machine learning models: A Logistic Regression Model and a Decision Tree Model in order to use the data to predict if an indiviudal has heart disease.
The data was also run through an exploratory data analysis to determine the coorelations between the features and the label. Through this, it was found that several variables
were irrelevant to the label such as Alochol Consumption (An outlier), An individuals race and individuals in excellent or great health.

The results showed that a logistic model was the best approach with a score of 91%, while the decision tree only scored an accuracy rating of 86%. Overall, these models can be imporved
through the expansion of the dataset to include more individuals with heart disease, and resolve outlier issues such as alocohol consumption 

Installation:
Once the repository has been downloaded, there are two python files to be run. Please ensure the dataset file is in the same folder as the python folders or it will not be able to
run. The file labelled "walji_noah_eda.py" will run the EDA analysis on the dataset. This will produce graphs that are also shown in the report. The console will also output probabilities 
to examine. Once you close the graph, the next one will open until all have run. The file labelled "walji+noah_model.py" will run both the Logistic and Decision tree model
It will also pre-process and clean the data as well as remove features. It will print out the final cleaned dataset preview and then run the models and print the accuracy.
It will also produce a graph for the logistic model and a tree for the decision tree. To proceed with the program please close these windows once done viewing.

About:
I am a 4th year double degree student in Buisness Administration and Computer Science at Laurier. I love to code and learn new langauges and libaries. Most of the time I work in the web development space,
working in both the front and back end. This project opened my eyes to many new tasks and libraries,
and allowed me to practice performing machine learning and AI techniques.

Licence:
GNU General Public License. Please see LICENCE.MD for more info

