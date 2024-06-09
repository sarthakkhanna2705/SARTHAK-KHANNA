# Hydraulic Health Monitoring


#Importing all the necessary libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr
# sklearn libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
# imbalanced-learn library
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
#joblib for model persistence
from joblib import dump, load

#Reading the data
hydraulic_systemdf = pd.read_csv('manufacturing_data.csv')
profile_data = pd.read_csv('profile.txt', sep="\t", header=None)
profile_data.columns = ['cooler_condition', 'valve_condition', 'internal_pump_leakage', 'hydraulic_accumulator', 'stable_flag']
#Data pre processing
hydraulic_systemdf = hydraulic_systemdf.drop(columns=["Unnamed: 0", "Time"])
hydraulic_systemdf = hydraulic_systemdf.rename(columns={
    "Cooling efficiency": "CE",
    "Cooling power": "CP",
    "Motor power W": "EPS1",
    "Volume flow l/min 1": "FS1",
    "Volume flow l/min 2": "FS2",
    "Pressure bar 1": "PS1",
    "Pressure bar 2": "PS2",
    "Pressure bar 3": "PS3",
    "Pressure bar 4": "PS4",
    "Pressure bar 5": "PS5",
    "Pressure bar 6": "PS6",
    "Efficiency factor": "SE",
    "Temperature 1": "TS1",
    "Temperature  2": "TS2",
    "Temperature  3": "TS3",
    "Temperature  4": "TS4",
    " Vibration mm/s": "VS1"
})
hydraulic_systemdf = pd.concat([hydraulic_systemdf,profile_data], axis=1)

hydraulic_systemdf.head(5)

#Explortory Data Analysis
hydraulic_systemdf.plot(x='Date',
              title = "Hydraulic Rig Target Features over Time",
              y=['cooler_condition', 'valve_condition', 'internal_pump_leakage', 'hydraulic_accumulator', 'stable_flag'],
              figsize=(10,8),
              subplots=True)
plt.tight_layout()
plt.savefig('target_features.png', format='png')
plt.show()

#Sensor Data Histogram
all_sensors = ['CE', 'CP', 'EPS1', 'FS1', 'FS2', 'PS1', 'PS2', 'PS3', 'PS4', 'PS5',
                  'PS6', 'SE', 'TS1', 'TS2', 'TS3', 'TS4', 'VS1']

#Creating subplots for each sensor
fig, axes = plt.subplots(nrows=len(all_sensors), figsize=(10, 30))

#Setting the title for the whole figure
fig.suptitle('Distribution of Sensors', fontsize=20, x=.53, y=1)

#Iterating over each sensor column and plot histogram
for i, column in enumerate(all_sensors):
    ax = axes[i]
    ax.hist(hydraulic_systemdf[column], bins=100)
    ax.set_title(column)
    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')

#Adjust spacing between subplots
plt.tight_layout()
plt.savefig('sensordistribution.png', format='png')
#Show the plot
plt.show()

# Create subplots for each sensor
fig, axes = plt.subplots(nrows=len(all_sensors), figsize=(10, 30))
for i, column in enumerate(all_sensors):
    ax = axes[i]
    ax.boxplot(hydraulic_systemdf[column])
    ax.set_title(column)
    ax.set_ylabel('Values')

# Adjust spacing between subplots
plt.tight_layout()
plt.savefig('sensorboxplot.png', format='png')
# Show the plot
plt.show()

# Subset the columns
profile_columns = ['cooler_condition','valve_condition', 'internal_pump_leakage', 'hydraulic_accumulator', 'stable_flag']

# Create subplots
fig, axes = plt.subplots(len(profile_columns), 1, figsize=(10, 10))

# Set the title for the whole figure
fig.suptitle('Class Distribution(Profile)', fontsize=20)

# Iterate over columns and plot count plots
for i, column in enumerate(profile_columns):
    ax = axes[i]
    sns.countplot(x=column, data=hydraulic_systemdf, ax=ax)
    ax.set_xlabel(column)
    ax.set_ylabel('Count')

    # Calculate total count of records for the current column
    total = len(hydraulic_systemdf[column])

    # Iterate over all bars and add percentage text inside each bar
    for p in ax.patches:
        height = p.get_height()
        # If height is 0, we want to avoid division by zero error
        if height == 0:
            continue
        percentage = f'{100 * height/total:.1f}%'
        ax.text(p.get_x()+p.get_width()/2., height/2, percentage, ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

# Adjust spacing between subplots
plt.tight_layout()

plt.savefig('class_distributionsprofile.png', format='png')

# Show the plot
plt.show()

#Co-relation Heatmap
# Select the columns for correlation heatmap
all_columns = ['SE', 'PS1', 'TS4', 'PS2', 'PS3', 'TS3', 'VS1', 'TS2', 'PS6', 'PS4', 'TS1',
           'PS5', 'CP', 'CE', 'EPS1', 'FS1', 'FS2', 'cooler_condition', 'valve_condition',
           'internal_pump_leakage', 'hydraulic_accumulator', 'stable_flag']

# Create a new figure with a size of 20x13
fig, ax = plt.subplots(figsize=(20, 13))

# Extract the selected columns and compute the correlation matrix
correlation_matrix = hydraulic_systemdf[all_columns].corr()

# Create a mask to hide the upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Plot a heatmap of the correlation matrix with the mask applied
sns.heatmap(correlation_matrix, annot=True, ax=ax, mask=mask)

# Set the title and rotate x-axis labels
ax.set_title('Correlation Heatmap', fontsize=20)
ax.set_xticklabels(ax.get_xticklabels(), rotation=-30, ha='left')

plt.savefig('correlation_heatmap.png', format='png')
# Show the plot
plt.show()

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sns.set(style="whitegrid")
target_variables = ['cooler_condition', 'valve_condition', 'internal_pump_leakage', 'hydraulic_accumulator', 'stable_flag']
for target in target_variables:

    y = hydraulic_systemdf[target]

    if target == "stable_flag":
        X = hydraulic_systemdf.drop(columns=['Date', 'stable_flag'])
    else:
        X = hydraulic_systemdf.drop(columns=['Date','cooler_condition', 'valve_condition', 'internal_pump_leakage', 'hydraulic_accumulator', 'stable_flag'])
        # Binary classification for stable_flag
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        classifiers = {
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'KNN': KNeighborsClassifier(),
            'SVM': SVC()
        }

        for name, model in classifiers.items():
            model.fit(X, y)

            # For tree-based models, plot feature importances
            if name in ['Decision Tree', 'Random Forest', 'Gradient Boosting']:
                importances = model.feature_importances_

                # Plot feature importances
                fig, ax = plt.subplots(figsize=(20, 6))
                color = sns.color_palette("viridis", len(importances))
                ax.bar(X.columns, importances, color=color)
                ax.set_xlabel('Features', fontsize=14)
                ax.set_ylabel('Importance Score', fontsize=14)
                ax.set_title(f"Feature Importances for {target} - {name}", fontsize=18)
                plt.xticks(rotation=-45, ha='left', fontsize=12)

                # Add grid lines
                ax.grid(axis='y', linestyle='--', alpha=0.7)

                # Save the figure as a png file
                plt.savefig("featureimportances.png", bbox_inches='tight')

                # Show the plot
                plt.show()


X_full_train, X_test = train_test_split(hydraulic_systemdf, test_size=0.2, random_state=1)
X_train, X_val = train_test_split(X_full_train, test_size=0.25, random_state=1)

def highlight_cell(value):
    # Check if the value is a string and ends with 'weighted avg'
    if isinstance(value, str) and value.endswith('weighted avg'):
        # Apply background color light blue and text color dark blue
        return 'background-color: #ADD8E6; color: #00008B'
    
    # Check if the value is a float and equal to the 'recall' value in the 'weighted avg' row
    if isinstance(value, float) and value == classification_rep_df.loc['weighted avg', 'recall']:
        # Apply background color light blue and text color dark blue
        return 'background-color: #ADD8E6; color: #00008B'
    
    # If none of the conditions match, no styling is applied
    return ''

import os
from joblib import dump

# Define the directory path
directory = 'models/'

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

target_var = ['cooler_condition', 'valve_condition', 'internal_pump_leakage', 'hydraulic_accumulator']

for target in target_var:
    # Prepare the training and validation sets
    X = X_train.drop(columns=['Date', 'cooler_condition', 'valve_condition', 'internal_pump_leakage', 'hydraulic_accumulator', 'stable_flag'])
    y = X_train[target]

    X_val_transformed = X_val.drop(columns=['Date', 'cooler_condition', 'valve_condition', 'internal_pump_leakage', 'hydraulic_accumulator', 'stable_flag'])
    y_val = X_val[target]

    # Create instances of your selected scaler and classifiers
    classifiers = {
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier()
    }

    for name, classifier in classifiers.items():
        # Create an instance of SMOTE inside the loop
        sm = SMOTE(random_state=1)

        # Create a pipeline with SMOTE, scaler, and classifier
        pipeline = ImbPipeline([('smote', sm), ('scaler', QuantileTransformer(output_distribution='normal', random_state=1)), ('classifier', classifier)])

        # Fit the pipeline on the training set
        X_resampled, y_resampled = sm.fit_resample(X, y)
        pipeline.fit(X_resampled, y_resampled)

        # Use the pipeline to make predictions on the validation set
        y_val_pred = pipeline.predict(X_val_transformed)

        # Compute precision, recall, and F1 score and output as a dictionary
        classification_rep = classification_report(y_val, y_val_pred, output_dict=True)

        # Convert the classification report dictionary to a DataFrame
        classification_rep_df = pd.DataFrame(classification_rep).transpose().drop(index=['accuracy', 'macro avg'])

        # Apply the custom styling function
        styled_df = classification_rep_df.style.applymap(highlight_cell).set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#424242'), ('color', '#f0f0f0')]},
            {'selector': 'tr:nth-child(odd)', 'props': [('background-color', '#424242'), ('color', '#f0f0f0')]},
            {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#303030'), ('color', '#f0f0f0')]}
        ])
        
        # Save the model to a file with a unique name
        dump(pipeline, f'{directory}{target}_{name}_model.joblib')

        # Display for checking
        print(f"Classification Report for {target} - {name}")
        display(styled_df)


from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from joblib import dump
import pandas as pd
#import dfi

# Define the columns to be one-hot encoded and the columns for the quantile transformer
categorical_features = ['cooler_condition', 'valve_condition', 'internal_pump_leakage', 'hydraulic_accumulator']
quant_features = ['CE', 'CP', 'EPS1', 'FS1', 'FS2', 'PS1', 'PS2', 'PS3', 'PS4', 'PS5','PS6', 'SE', 'TS1', 'TS2', 'TS3', 'TS4', 'VS1']

# Create a preprocessor that applies the OneHotEncoder and the QuantileTransformer to the specified columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', QuantileTransformer(output_distribution='normal', random_state=1), quant_features),
        ('cat', OneHotEncoder(), categorical_features)])

# Create an instance of SMOTE
sm = SMOTE(random_state=1)

# Create a pipeline with SMOTE, preprocessor, and classifier
pipeline = ImbPipeline([('smote', sm), ('preprocessor', preprocessor), ('classifier', classifier)])

# Drop unnecessary columns from X_train and X_val
X = X_train.drop(columns=['Date', 'stable_flag'])
y = X_train['stable_flag']

X_val_transformed = X_val.drop(columns=['Date', 'stable_flag'])
y_val = X_val['stable_flag']

# Fit the pipeline on the training set
pipeline.fit(X, y)

# Transform X_val using the preprocessor
X_val_transformed = pipeline.named_steps['preprocessor'].transform(X_val_transformed)

# Use the pipeline to make predictions on the validation set
y_val_pred = pipeline.named_steps['classifier'].predict(X_val_transformed)

# Compute precision, recall, and F1 score and output as a dictionary
classification_rep = classification_report(y_val, y_val_pred, output_dict=True)

# Convert the classification report dictionary to a DataFrame
classification_rep_df = pd.DataFrame(classification_rep).transpose().drop(index=['accuracy', 'macro avg'])

# Apply the custom styling function
styled_df = classification_rep_df.style.applymap(highlight_cell).set_table_styles([
    {'selector': 'th', 'props': [('background-color', '#424242'), ('color', '#f0f0f0')]},
    {'selector': 'tr:nth-child(odd)', 'props': [('background-color', '#424242'), ('color', '#f0f0f0')]},
    {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#303030'), ('color', '#f0f0f0')]}
])

# Save the styled DataFrame as a png image
#dfi.export(styled_df, 'images/stable_flag_class_rep.png')

# Save the model to a file
dump(pipeline, 'models/stable_flag_model.joblib')

# Display for checking
print(f"Classification Report: stable_flag")
display(styled_df)


X_train_drop = X_train.drop(columns=['Date','cooler_condition', 'valve_condition', 'internal_pump_leakage', 'hydraulic_accumulator', 'stable_flag'])
X_val_drop = X_val.drop(columns=['Date','cooler_condition', 'valve_condition', 'internal_pump_leakage', 'hydraulic_accumulator', 'stable_flag'])
X_test_drop = X_test.drop(columns=['Date','cooler_condition', 'valve_condition', 'internal_pump_leakage', 'hydraulic_accumulator', 'stable_flag'])


# Get feature names from the preprocessor
num_features = ['CE','CP','EPS1','FS1','FS2','PS1','PS2','PS3','PS4','PS5','PS6','SE','TS1','TS2','TS3','TS4', 'VS1']
cat_features = ['cooler_condition','valve_condition','internal_pump_leakage', 'hydraulic_accumulator']

# Numeric features remain the same
feature_names = num_features.copy()

# Get the feature importances from the trained model
important = stable_flag_model.named_steps['classifier'].feature_importances_

# For categorical features, we need to append the encoded feature names
ohe = stable_flag_model.named_steps['preprocessor'].named_transformers_['cat']
feature_names.extend(ohe.get_feature_names_out(cat_features))

# At this point, feature_names contains original numeric feature names and the transformed categorical feature names

# Now you can use this in your plot
fig, ax = plt.subplots(figsize=(22,6))
ax.bar(feature_names, important)
ax.set_xlabel('Features')
ax.set_ylabel('Importance Score')
ax.set_title(f"Feature Importances for {target}")
plt.xticks(rotation=-30, ha='left')
plt.subplots_adjust(bottom=0.3) 
plt.savefig('test_feature_importance.png', format='png')
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from joblib import load
import pandas as pd

# Assuming you have the classification report stored in classification_rep_df
# Replace this with the actual classification report DataFrame
classification_rep_df = pd.DataFrame(classification_rep).transpose().drop(index=['accuracy', 'macro avg'])

# Create a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(classification_rep_df.iloc[:, :3], annot=True, cmap='viridis', fmt='.2f', linewidths=.5, cbar=False)
plt.title("Classification Report Heatmap")
plt.xlabel("Metrics")
plt.ylabel("Degradation States")
plt.show()

# Extract metrics for plotting
metrics_to_plot = ['precision', 'recall', 'f1-score']
metrics_df = classification_rep_df[metrics_to_plot]

# Plot the bar plot
metrics_df.plot(kind='bar', figsize=(12, 6), colormap='viridis')
plt.title("Classification Report Metrics for Stable Flag Prediction")
plt.xlabel("Degradation States")
plt.ylabel("Score")
plt.legend(title="Metrics", loc='upper right')
plt.xticks(rotation=45, ha='right')
plt.show()



# Drop unnecessary columns from X_train and X_val
X = X_train.drop(columns=['Date','cooler_condition', 'valve_condition', 'internal_pump_leakage', 'hydraulic_accumulator', 'stable_flag'])
y = X_train['valve_condition']

X_val_transformed = X_val.drop(columns=['Date','cooler_condition', 'valve_condition', 'internal_pump_leakage', 'hydraulic_accumulator', 'stable_flag'])
y_val = X_val['valve_condition']

# Create an instance of SMOTE
sm = SMOTE(random_state=1)

# Create instances of your selected scaler and classifier
quant = QuantileTransformer(output_distribution='normal', random_state=1)
classifier = RandomForestClassifier(random_state=1)  

# Create a pipeline with SMOTE, scaler, and classifier
pipeline = ImbPipeline([('smote', sm), ('scaler', quant), ('classifier', classifier)])

# Fit the pipeline on the training set
pipeline.fit(X, y)

# Transform X_val using the scaler
X_val_transformed = pipeline.named_steps['scaler'].transform(X_val_transformed)

# Use the pipeline to make predictions on the validation set
y_val_pred = pipeline.named_steps['classifier'].predict(X_val_transformed)

# Compute precision, recall, and F1 score
classification_rep = classification_report(y_val, y_val_pred)
print("Classification Report:\n", classification_rep)

# Define the parameter grid
param_grid = {
    'classifier__n_estimators': [100, 200, 300, 500],
    'classifier__max_depth': [None, 5, 10, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['auto', 'sqrt', 'log2'],
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__class_weight': [None, 'balanced', 'balanced_subsample']
}

# Create RandomizedSearchCV object
random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, scoring='precision_weighted', verbose=1, n_iter=10, cv=5)

# Fit the random search to the training data
random_search.fit(X, y)

# Print the best parameters
print("Best parameters: ", random_search.best_params_)

# Use the best model to make predictions on X_val
best_model = random_search.best_estimator_
y_val_pred = best_model.predict(X_val_transformed)

# Compute precision, recall, and F1 score
classification_rep = classification_report(y_val, y_val_pred)
print("Classification Report:\n", classification_rep)

