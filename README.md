# MACHINE-LEARNING-MODEL-IMPLEMENTATION

*COMPANY:* CodTech IT Solutions

*NAME:* A Jenita Roselin

*INTERN ID:* CT08FYS

*DOMAIN:* Python Programming

*DURATION:* 4 weeks

*MENTOR:* Neela Santhosh Kumar

# Task Description: Machine Learning Implementation- Spam Email Detection

# *1. Importing Required Libraries*

In this section, essential libraries are imported to facilitate data manipulation, machine learning, and file handling. NumPy and Pandas are fundamental libraries for data analysis, where NumPy handles array-based operations, and Pandas is used for managing and analyzing tabular data. Scikit-learn is a library for machine learning, which includes tools for creating, training, and evaluating models. Here, train_test_split is used for splitting the data into training and test sets, while LogisticRegression builds the model for classifying spam and ham emails. accuracy_score is used to evaluate the performance of the model, and TfidfVectorizer is responsible for converting text data into numerical format. Kagglehub allows the user to directly download datasets from Kaggle, and os helps in managing file paths for dataset access.


# *2. Downloading the Dataset*

The program uses kagglehub to download the dataset. 

Kaggle is a popular platform for machine learning datasets, and this specific dataset contains labeled spam and ham emails. By using the dataset_download() method, the dataset is fetched directly from Kaggle. The path where the dataset is stored on the local machine is captured in the variable path, which will be used in the next steps for loading the actual data.


# *3. Defining File Path*

Once the dataset is downloaded, the code constructs the full file path to the CSV file using the os.path.join() method. The file path (file_path) holds the path that containg the CSV file which in turn contains the spam/ham data for easy referencing in upcoming steps.


# *4. Loading the Dataset*

The dataset is loaded into a Pandas DataFrame using the read_csv() function. This method reads the CSV file, which contains the email data and labels, and converts it into a structured format that can be easily manipulated and analyzed. By loading the data into a DataFrame, the code prepares it for the subsequent data processing and model training steps.


# *5. Handling Missing Data*

Data often contains missing or null values, which can interfere with the model's performance. The code addresses this by using the where() function from Pandas to replace any missing data with empty strings. This ensures that the model can handle the data without issues, as missing values are typically problematic in machine learning algorithms.


# *6. Modifying Label Values*

The dataset contains labels 0 for spam and 1 for ham, but these values are adjusted for clarity. Initially, spam is marked as 0 and ham as 1. The code changes 0 to 9, then 1 to 0, and finally converts 9 back to 1. This re-labeling process ensures that the dataset is more comprehensible, making it easier to understand the classification task.


# *7. Separating Features and Labels*

The dataset is now divided into features (X) and labels (Y). The feature X contains the email text, which will be used as input for the model, while Y contains the labels (spam or ham), which are the target values the model will predict. This separation is essential for training the model.


# *8. Splitting Data into Training and Test Sets*

To evaluate the model's performance effectively, the dataset is split into a training set and a test set using the train_test_split function. The training set (80% of the data) is used to train the model, while the test set (20% of the data) is used to evaluate its performance. This split ensures that the model is tested on data it has never seen before, providing a more accurate representation of how it will perform in real-world scenarios.


# *9. Feature Extraction (Converting Text to Numerical Format)*

The TfidfVectorizer is used to convert the email text (which is in string format) into numerical values that can be processed more easily by the model. TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical method used to evaluate the importance of a word in a document relative to the entire dataset. By applying fit_transform() to the training set and transform() to the test set, the text data is converted into a matrix of numerical values. This step is crucial because machine learning models can only work with numerical data.


# *10. Training the Logistic Regression Model*

A Logistic Regression model is chosen for this classification task because it is simple and effective for binary classification problems like distinguishing between spam and ham. The model is trained using the numerical features from the training set (X_train_features) and the corresponding labels (Y_train). The fit() function allows the model to learn the relationship between the features (email content) and the labels (spam or ham).


# *11. Evaluating the Model's Performance*

Once the model is trained, its performance is evaluated using both the training and test sets. The accuracy_score function calculates how well the model's predictions match the actual labels. The model's predictions on both the training set and the test set are compared to the true labels, and the accuracy is calculated for both. This step allows us to determine how well the model is performing and whether it is overfitting or underfitting.


# *12. Real-Time Email Classification*
    
This part allows the user to input a new email, and the model predicts whether it is spam or ham. The email text is transformed into the same numerical format used for training the model using TfidfVectorizer. After transformation, the model makes a prediction about the email's classification. This feature allows the model to be used in a real-world application where users can input emails and get immediate feedback on whether they are spam or ham.


# *13. Displaying the Result*

The model's prediction is then displayed to the user. If the prediction is 1, it indicates that the email is classified as ham (non-spam), and the program prints "Ham mail." If the prediction is 0, it indicates that the email is classified as spam, and "Spam mail" is printed instead. This provides an interactive and user-friendly way to classify emails.

# Ouput Screenshot:








