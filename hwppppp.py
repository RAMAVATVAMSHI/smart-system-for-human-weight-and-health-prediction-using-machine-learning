import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score  # Make sure GridSearchCV is imported
import pickle

# Constants
BMI_CATEGORIES = ['Underweight', 'Normal', 'Overweight', 'Obese']
FOOD_HABITS = ['Junk Food', 'Vegetarian', 'Vegan', 'Balanced']

# Load dataset (make sure the path is correct)
try:
    df = pd.read_csv(r"C:\Users\rama\OneDrive\Documents\hhd mini project.zip\hwp") 
except FileNotFoundError:
    print("Error: 'humanweight1.csv' not found. Please check the file path.")
    exit()

# Trim whitespace from column names
df.columns = df.columns.str.strip()

# Verify if 'BMI' and 'BMI Category' exist in the DataFrame
if 'bmi' not in df.columns or 'bmi_cat' not in df.columns:  # Corrected column names
    print("Error: 'bmi' and/or 'bmi_cat' columns are missing in the dataset.")  # Corrected column names
    exit()

# Function to categorize BMI
def categorize_bmi(bmi):
    if bmi < 18.5:
        return BMI_CATEGORIES[0]  # Underweight
    elif 18.5 <= bmi < 25:
        return BMI_CATEGORIES[1]  # Normal
    elif 25 <= bmi < 30:
        return BMI_CATEGORIES[2]  # Overweight
    else:
        return BMI_CATEGORIES[3]  # Obese

# Function to suggest diet
def suggest_diet(bmi_category):
    if bmi_category == BMI_CATEGORIES[0]:  # Underweight
        return "You should focus on consuming more calorie-dense foods and incorporate strength training exercises."
    elif bmi_category == BMI_CATEGORIES[1]:  # Normal
        return "Maintain a balanced diet with a mix of carbohydrates, proteins, and healthy fats. Regular exercise is important for overall health."
    elif bmi_category == BMI_CATEGORIES[2]:  # Overweight
        return "Try to reduce your calorie intake by cutting back on high-calorie and processed foods. Increase your physical activity and aim for a gradual weight loss."
    elif bmi_category == BMI_CATEGORIES[3]:  # Obese
        return "You should focus on portion control, reducing intake of high-calorie foods, and increasing physical activity. Consult with a healthcare professional for a personalized weight loss plan."

# Function to recommend exercise
def recommend_exercise(bmi_category):
    if bmi_category == BMI_CATEGORIES[0]:  # Underweight
        return "Strength training exercises to build muscle mass."
    elif bmi_category == BMI_CATEGORIES[1]:  # Normal
        return "A combination of cardiovascular exercises (like walking, jogging, or swimming) and strength training."
    elif bmi_category == BMI_CATEGORIES[2]:  # Overweight
        return "Cardiovascular exercises like running, cycling, or aerobics combined with strength training."
    elif bmi_category == BMI_CATEGORIES[3]:  # Obese
        return "Low-impact exercises such as walking, swimming, or cycling to start, then gradually increase intensity and duration."

# Function to calculate BMI
def calculate_bmi(weight, height):
    return weight / ((height / 100) ** 2)

# Encoding categorical variables (using dictionaries - Recommended)
gender_mapping = {'Male': 0, 'Female': 1}
food_habits_mapping = {'Junk Food': 0, 'Vegetarian': 1, 'Vegan': 2, 'Balanced': 3}
smoking_mapping = {'Yes': 0, 'No': 1}
alcohol_mapping = {'Never': 0, 'Rarely': 1, 'Occasionally': 2, 'Regular Drinker': 3}
exercise_mapping = {'None': 0, 'Irregular': 1, 'Regular': 2}

df['Gender'] = df['Gender'].map(gender_mapping)
df['Food_Habits'] = df['Food_Habits'].map(food_habits_mapping)
df['Smoking'] = df['Smoking'].map(smoking_mapping)
df['Alcohol_Consumption'] = df['Alcohol_Consumption'].map(alcohol_mapping)
df['Exercise_Routines'] = df['Exercise_Routines'].map(exercise_mapping)

# Split the data into features and target
X = df.drop(['bmi', 'bmi_cat'], axis=1)
y = df['bmi_cat']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# --- Hyperparameter Tuning and Cross-Validation (Corrected) ---
param_grid = {  # Define the parameter grid
    'max_depth': [None, 3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5]
}

# Create GridSearchCV object and *FIT* it to the data
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='accuracy')
grid_search.fit(X_train, y_train)  # This is the CRUCIAL line that was missing

best_classifier = grid_search.best_estimator_  # Now grid_search is defined

best_classifier = grid_search.best_estimator_  # Assuming you used GridSearchCV

# Save the model (use best_classifier)
with open('bmi_classifier.pkl', 'wb') as file:
    pickle.dump(best_classifier, file)  # Correct: best_classifier is now defined

# ... (Later, when loading)
with open('bmi_classifier.pkl', 'rb') as file:
    loaded_classifier = pickle.load(file) # Load it as loaded_classifier


# ***CREATE AND TRAIN THE CLASSIFIER HERE (Correct Location)***
classifier = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=10, min_samples_leaf=5)
classifier.fit(X_train, y_train)

# Predict the target labels on the testing set
y_pred = classifier.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(conf_matrix )

# Function to get user input (remains exactly the same)
def get_user_input():
    while True:
        try:
            age = int(input("Enter your age: "))
            if age <= 0 or age > 120:
                raise ValueError("Please enter a valid age between 1 and 120.")

            gender = input("Enter your gender (Male/Female): ").capitalize()
            if gender not in ['Male', 'Female']:
                raise ValueError("Please enter 'Male' or 'Female'.")

            height_cm = int(input("Enter your height in cm: "))  # Corrected prompt
            if height_cm <= 0 or height_cm > 300:
                raise ValueError("Please enter a valid height.")

            weight = float(input("Enter your weight in kg: "))
            if weight <= 0 or weight > 500:
                raise ValueError("Please enter a valid weight.")

            food_habits = input("Enter your food habits (Junk Food/Vegetarian/Vegan/Balanced): ").capitalize()
            if food_habits not in [habit.capitalize() for habit in FOOD_HABITS]:  # Case-insensitive check
                raise ValueError("Please enter a valid food habit.")
            smoking = input("Do you smoke? (Yes/No): ").capitalize()
            if smoking not in ['Yes', 'No']:
                raise ValueError("Please enter 'Yes' or 'No'.")

            alcohol_consumption = input("Do you drink alcohol? (Never/Rarely/Occasionally/Regular Drinker): ").capitalize()
            if alcohol_consumption not in ['Never', 'Rarely', 'Occasionally', 'Regular Drinker']:
                raise ValueError("Please enter a valid alcohol consumption habit.")

            exercise_routines = input("Enter your exercise routines (None/Irregular/Regular): ").capitalize()
            if exercise_routines not in ['None', 'Irregular', 'Regular']:
                raise ValueError("Please enter a valid exercise routine.")

            return age, gender, height_cm, weight, food_habits, smoking, alcohol_consumption, exercise_routines
        except ValueError as ve:
            print("Error:", ve)

# Get user input
user_input = get_user_input()
age, gender, height_cm, weight, food_habits, smoking, alcohol_consumption, exercise_routines = user_input
# Encode user input (using the mapping dictionaries)
user_data = {
    'Age': age,
    'Gender': gender_mapping[gender],  # Use gender directly
    'Height': height_cm,
    'Weight': weight,
    'Food_Habits': food_habits_mapping[food_habits],  # Use food_habits directly
    'Smoking': smoking_mapping[smoking],  # Use smoking directly
    'Alcohol_Consumption': alcohol_mapping[alcohol_consumption],  # Use alcohol_consumption directly
    'Exercise_Routines': exercise_mapping[exercise_routines]  # Use exercise_routines directly
}

user_df = pd.DataFrame([user_data])

# Predict BMI category based on user input
try:
    predicted_bmi_category = classifier.predict(user_df.drop('Weight', axis=1))[0]
except ValueError as e:
    print(f"Error during prediction: {e}")
    print("Check if your training data and user input have the same features (excluding 'Weight').")
    exit()

# Calculate BMI based on user weight and height
bmi = calculate_bmi(weight, height_cm)
predicted_bmi_category = loaded_classifier.predict(user_df.drop('Weight', axis=1))[0]  # Get the predicted category


# Suggest diet and exercise based on predicted BMI category
diet_recommendation = suggest_diet(predicted_bmi_category)
exercise_recommendation = recommend_exercise(predicted_bmi_category)

# Print the results (formatted like in your image)
print("\n")  # Add an extra newline for better spacing
print(f"Enter your height in cm: {height_cm}")
print(f"Enter your food habits (Junk Food/Vegetarian/Vegan/Balanced): {food_habits}")
print(f"Do you smoke? (Yes/No): {smoking}")
print(f"Do you drink alcohol? (Never/Rarely/Occasionally/Regular Drinker): {alcohol_consumption}")
print(f"Enter your exercise routines (None/Irregular/Regular): {exercise_routines}")
print(f"\nPredicted BMI Category: {predicted_bmi_category}")
print(f"\nDiet Recommendation: {diet_recommendation}")
print(f"\nExercise Recommendation: {exercise_recommendation}")
