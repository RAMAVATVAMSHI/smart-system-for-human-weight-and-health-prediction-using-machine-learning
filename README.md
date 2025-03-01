# smart-system-for-human-weight-and-health-prediction-using-machine-learning
Overview
This project aims to classify individuals into different BMI categories based on their personal attributes and health habits. The classification is performed using a Decision Tree Classifier, and the model is trained on a dataset containing various health-related features.

Files Included
hwppppp.py: The main Python script that contains the implementation of the BMI classification model, including data preprocessing, model training, and user interaction.
humanweight1.csv: A CSV file containing the dataset used for training the model. It includes features such as age, gender, height, weight, food habits, smoking status, alcohol consumption, exercise routines, and the corresponding BMI category.
humanweight1.xlsx: An Excel file that may contain similar data as the CSV file but is not utilized in the current implementation.
Dataset Description
The dataset consists of the following columns:

Age: Age of the individual (in years)
Gender: Gender of the individual (Male/Female)
Height: Height of the individual (in cm)
Food_Habits: Dietary habits (Junk Food/Vegetarian/Vegan/Balanced)
Smoking: Smoking status (Yes/No)
Alcohol_Consumption: Frequency of alcohol consumption (Never/Rarely/Occasionally/Regular Drinker)
Exercise_Routines: Exercise frequency (None/Irregular/Regular)
bmi_cat: BMI category (Underweight/Normal/Overweight/Obese)
bmi: Calculated BMI value
Key Features
BMI Calculation: The script includes a function to calculate BMI based on weight and height.
Diet and Exercise Recommendations: Based on the predicted BMI category, the script provides personalized diet and exercise recommendations.
Model Training: The Decision Tree Classifier is trained using hyperparameter tuning with GridSearchCV for optimal performance.
User Input: The script prompts the user for their personal information and health habits to predict their BMI category.
Usage Instructions
Ensure that the required libraries are installed:

numpy
pandas
scikit-learn
pickle
You can install these libraries using pip:

bash
Run
Copy code
pip install numpy pandas scikit-learn
Place the humanweight1.csv file in the specified path in the script or update the path in the script accordingly.

Run the hwppppp.py script:

bash
Run
Copy code
python hwppppp.py
Follow the prompts to enter your personal information.

The script will output your predicted BMI category along with diet and exercise recommendations.

Evaluation Metrics
The model's performance is evaluated using the following metrics:

Accuracy: The proportion of correct predictions.
Precision: The ratio of true positive predictions to the total predicted positives.
Recall: The ratio of true positive predictions to the total actual positives.
F1 Score: The harmonic mean of precision and recall.
Confusion Matrix: A table used to describe the performance of the classification model.
Conclusion
This project provides a comprehensive approach to BMI classification and offers personalized health recommendations based on user input. The use of machine learning techniques allows for accurate predictions and insights into individual health behaviors.
