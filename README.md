# SOC-midterm-review
## Python Basics

Learning Python has been a very useful experience for me. As a beginner, I started with no prior knowledge, but by following tutorials, reading online resources, and practicing regularly, I have gained a good understanding of the basics. Below is a summary of the key topics I have covered so far.

### Getting Started

I began by installing Python 3 in Google Colab and done all my coding on colab only.

### About Python

Python is designed to be easy to read and write. It does not use symbols like curly braces or semicolons. Instead, it uses indentation to structure the code, which makes the code cleaner and more readable.

### Variables and Data Types

I learned that Python uses dynamic typing, which means I don’t need to declare variable types manually. Some common data types I practiced with include:

* Integers (e.g., 5)
* Floats (e.g., 3.14)
* Strings (e.g., "hello")
* Booleans (True or False)

I also learned how to convert between data types using functions like `int()`, `float()`, and `str()`.

### Working with Strings

Strings are used to store and work with text. I learned how to:

* Join strings (concatenation)
* Access parts of strings (slicing and indexing)
* Use string methods like `.lower()`, `.upper()`, `.replace()`, and `.find()`
* Format strings using f-strings and the `.format()` method

### Input and Output

Using the `input()` function, I was able to collect user input and use it in my programs. This made my programs more interactive and flexible.

### Lists, Tuples, and Dictionaries

I studied three main data structures:

* **Lists**: Ordered and changeable. I used methods like `.append()`, `.remove()`, and slicing.
* **Tuples**: Ordered but not changeable. Useful for fixed collections.
* **Dictionaries**: Store key-value pairs. Very helpful for fast lookups and organizing data.

### Control Flow: Conditionals and Loops

Control flow allows programs to make decisions and repeat actions. I practiced using:

* `if`, `elif`, and `else` for making decisions
* `for` and `while` loops for repeating tasks
* List comprehensions to create lists in a short and efficient way

### Functions

Functions are blocks of code that can be reused. I learned how to define them using `def`, pass parameters, and return values. Functions made my code more organized and easier to read.

### Error Handling and Comments

To manage errors, I used `try` and `except` blocks. This helped prevent programs from crashing. I also used comments (written with `#`) to explain my code and make it easier to understand.

### Modules and Libraries

I learned how to use Python’s built-in modules and some external libraries. I used `import` to bring them into my programs. Some examples include:

* `math` for mathematical operations
* `random` for generating random numbers

I also started learning about:

* `NumPy` for working with arrays and numerical data
* `Pandas` for handling and analyzing data
* `Matplotlib` for creating graphs and visualizations
* `Scikitlearn` for making ML Models like Linear Regression and Logistic Regression 

### Working with Files

I practiced reading from and writing to files using `open()`, `.read()`, `.write()`, and `.close()`. This is useful for storing data or working with external text files.

### Practical Applications

Python can be used in many fields. Some examples I learned about include:

* Web development
* Automation and scripting
* Data analysis and visualization
* Machine learning and artificial intelligence
* Quick software development and prototyping

## Machine Learning Basics ##

Machine Learning (ML) is a part of Artificial Intelligence. It helps computers to learn from data and make decisions or predictions without writing rules for every situation. After watching some tutorials and reading articles, I now understand the main ideas and how ML works in simple terms.

### What is Machine Learning?

Machine learning is when a computer learns patterns from old data and uses that to guess or decide something on new data. The computer gets better as it sees more data, just like people learn from experience. We can see machine learning in things like YouTube recommendations, voice assistants, or even detecting fraud in banks.

### Main Parts in Machine Learning

Every ML project needs a few important things:

* **Data** – All the information we use. It can be text, numbers, images, etc.
* **Features** – The important details from the data used to make predictions.
* **Model** – A formula or system that connects features to results.
* **Training** – Teaching the model using the data.
* **Evaluation** – Checking how good the model is by using scores like accuracy.
* **Optimization** – Making the model better by changing settings to reduce mistakes.

### Types of Machine Learning

There are 4 main types of ML:

| Type                     | What it does                                         | Example                        |
| ------------------------ | ---------------------------------------------------- | ------------------------------ |
| Supervised Learning      | Learns from data with answers already given.         | Spam detection, image labeling |
| Unsupervised Learning    | Finds patterns in data with no answers.              | Customer groups, clustering    |
| Reinforcement Learning   | Learns by doing and getting rewards or penalties.    | Game bots, robots              |

### ML Workflow – How It Happens

1. **Collect and clean data** – Get good quality data.
2. **Understand the data** – Use charts and graphs to see what’s inside.
3. **Feature engineering** – Pick and change the useful data parts.
4. **Choose model** – Pick the algorithm (like regression or classification).
5. **Train model** – Show the model the training data.
6. **Tune the model** – Try different settings to improve results.
7. **Evaluate** – Test if it’s working well using accuracy and other scores.
8. **Deploy** – Use the model in real life applications.

### Important Algorithms

#### 1. Linear Regression

This is a basic supervised learning method. It predicts numbers (not categories). It draws a straight line to show the relationship between input and output.

* **Formula**:
  `y = mx + b`

* **Used for**:
  House prices, sales prediction, trends

* **Things I learned**:

  * Mean Squared Error (to see mistake size)
  * Gradient Descent (to improve model)
  * R-squared, RMSE (to check how good the model is)

#### 2. Logistic Regression

This is used for classification (like yes or no). It predicts chances and uses something called sigmoid function.

* **Formula**:
  `P(y=1|x) = 1 / (1 + e^-(mx + b))`

* **Used for**:
  Spam email detection, health predictions

* **Important points**:

  * Decision boundary
  * Log loss (error score)
  * Accuracy, precision, recall, confusion matrix

### How to Check Model Performance

To know how well the model works, we use:

* **Accuracy** – How many answers were right
* **Precision & Recall** – Good for unbalanced data
* **F1 Score** – Mix of precision and recall
* **ROC Curve / AUC** – How well the model tells classes apart

### Where Machine Learning Is Used

ML is used in many places, like:

* Recommending movies or music
* Catching fraud in banks
* Diagnosing health problems
* Self-driving cars
* Chatbots and language tools

## Basics of Neural Networks

Neural networks are an important part of artificial intelligence (AI) and machine learning. They are made to work like the human brain, so computers can learn from data and make smart decisions. I watched some YouTube videos and read a few articles to understand how they work, summary of same is as follows:

### What is a Neural Network?

A neural network is a computer system that tries to copy how our brain works. It learns patterns from data, so it can make predictions or solve problems. For example, it can be used for things like recognizing faces in photos, translating languages, or playing games.

### Main Parts of a Neural Network

Neural networks are made of small units called *neurons*. These neurons are connected together in layers:

* **Input Layer** – This is where the data comes in. For example, the pixel values of a picture.
* **Hidden Layer(s)** – These layers do the calculations and help find patterns in the data.
* **Output Layer** – This layer gives the final result, like a prediction or answer.

Other important parts are:

* **Weights and Biases** – These are numbers that help decide how important each input is.
* **Activation Functions** – These are used to help the network understand complex things. Some examples are:

  * **Sigmoid** – Gives results between 0 and 1.
  * **ReLU** – Makes all negative numbers zero, keeps positive numbers the same.
  * **Tanh** – Gives results between -1 and 1.

### How Neural Networks Work

Here is how a neural network usually works:

1. **Feedforward** – Data goes from the input layer, through the hidden layers, to the output layer.
2. **Prediction** – The output layer gives a prediction or result.
3. **Training with Backpropagation** – If the prediction is wrong, the network fixes its mistakes by changing the weights and biases. This happens many times so it can learn better.

### Types of Neural Networks

* **Single Layer Perceptron** – This is the simplest kind. It only has one layer between input and output. Good for simple problems.
* **Multi-Layer Perceptron (MLP)** – Has more than one hidden layer. Can solve harder problems.
* **Deep Neural Networks** – These have many hidden layers. They are used for very complex tasks, like speech or image recognition.

### Why Neural Networks Are Important

Neural networks are powerful because they can learn from data without needing humans to tell them every rule. They are used in many areas like:

* Image and voice recognition
* Chatbots and translation
* Self-driving cars
* Games like chess or Go
* Medical tools that help doctors

They are very useful because they can understand things directly from raw data, like pictures or sound, without too much help from humans.

Sure! Here’s a more beginner-friendly and simple version of your report on Convolutional Neural Networks (CNNs), written in plain English like a fresher might explain:

## Basics of Convolutional Neural Networks (CNNs)

Convolutional Neural Networks, or CNNs, are a special type of deep learning model mostly used for images. They are very good at understanding pictures, and are used in things like recognizing faces, detecting objects, and classifying images (like telling if a photo has a cat or a dog).

### What Makes CNNs Different?

Normal neural networks look at data as a flat list of numbers. But CNNs keep the shape of the image (like width and height) while working on it. This helps CNNs find patterns in images, like edges, shapes, or textures, directly from the raw image without needing a human to point them out.

### Main Parts of a CNN

#### 1. **Convolutional Layer**

* This is the main part of a CNN.
* It uses small filters (like a window) that move across the image.
* Each filter looks for patterns in the image and creates a new image called a feature map.
* You can control how it moves (stride), and how much of the image it sees (filter size and padding).

#### 2. **Activation Function**

* After filtering, we use an activation function to add non-linearity (so the model can learn complex things).
* The most common one is ReLU, which replaces negative numbers with 0.

#### 3. **Pooling Layer**

* This layer makes the data smaller so the model is faster and more stable.
* Max pooling is common — it keeps the biggest number from each small area.
* It helps to focus on the most important parts of the image.

#### 4. **Fully Connected Layer (Dense Layer)**

* After all the filters and pooling, the data is turned into a list.
* Then it goes through one or more dense layers to make the final prediction (like telling if the image is of a dog or a cat).

### How CNNs Work (Step-by-Step)

1. **Input:** The image goes into the network.
2. **Feature Extraction:** CNN looks at small parts of the image to find things like lines, shapes, or patterns.
3. **Classification:** In the end, the model uses what it found to decide what’s in the image.

### Why CNNs Use Convolutions

* Convolutions look at small areas of the image, so the model can understand local features.
* The same filters are used again and again across the image, which saves memory and time.
* Lower layers find simple patterns like edges, and higher layers build on that to find more complex shapes.

### A Simple CNN Architecture

A basic CNN has:

* Input layer
* One or more groups of:

  * Convolution → Activation → Pooling
* Then:

  * Flatten → Dense Layer → Output

Famous models like LeNet, AlexNet, and ResNet are built using this structure with more improvements.

### Where CNNs Are Used

CNNs are used in many fields, such as:

* Image classification (cat or dog)
* Object detection (where is the object in the image)
* Face recognition
* Medical imaging (like finding problems in X-rays)
* Self-driving cars (seeing and understanding the road)
* Changing photo styles (like turning a photo into art)

## Deep Learning?

Deep learning is a type of machine learning that uses **neural networks with many layers** (this is why it’s called "deep"). Instead of writing rules, deep learning lets the computer figure things out on its own. These models can learn patterns from big data, like photos, sounds, or text. Deep learning is used in things like face recognition, voice assistants, translations, and more.

### How Is Deep Learning Different from Regular Machine Learning?

* In **normal ML**, people have to tell the computer what features (data points) to focus on.
* In **deep learning**, the computer learns these features by itself, without much human help. This makes it better for complicated problems like images or speech.

### Main Concepts I Learned

#### 1. **Forward and Backward Pass**

* **Forward pass** – data goes through the layers to make a prediction.
* **Loss function** – checks how wrong the prediction is.
* **Backpropagation** – the model learns from the mistake by adjusting weights to improve next time.

#### 2. **Training Terms**

* **Epoch** – one full round over all the training data.
* **Batch** – a small group of data used in one step.
* **Iteration** – one update of the model after a batch.

#### 3. **Overfitting and Underfitting**

* **Overfitting** – model learns training data too well and does badly on new data.
* **Underfitting** – model doesn’t learn enough, so it does badly on both old and new data.

### Why Deep Learning Works So Well

* It can **learn from raw data** without needing humans to pick the features.
* It works well with **big data** and can learn very complex things.
* Each layer **builds on top of the last one**, so it can learn simple to advanced patterns (like lines → shapes → faces).

### Where Deep Learning Is Used

Deep learning is behind many cool technologies:

* Image and voice recognition
* Chatbots and language translation
* Self-driving cars
* Medical diagnosis
* Recommendation systems (like Spotify or YouTube)
* Detecting fraud
* Game playing (like AlphaGo)

### Challenges in Deep Learning

* It needs **a lot of data** to work well.
* It needs **strong computers**, like GPUs, to train faster.
* Sometimes the models are like **black boxes** – it’s hard to understand how they make decisions.

Sure! Here's a simplified and more human version of your Scikit-learn report — written in plain, beginner-style English, like how a fresher or someone new to machine learning might explain what they’ve learned:


## Scikit-learn?

Scikit-learn (or `sklearn`) is a free and open-source Python library. It’s built on other Python tools like NumPy, SciPy, and Matplotlib. It’s made for both beginners and experienced users to easily do tasks in machine learning like classification, regression, and clustering.

### Why It’s Useful (Main Features)

* **Same Method for All Models**
  Most models use `.fit()`, `.predict()`, and `.score()` — once you learn it for one model, you can apply it to others.

* **Good Documentation**
  The official website explains things clearly with examples, so it’s easy to follow.

* **Many Algorithms**
  You can use it for classification, regression, clustering, and even dimensionality reduction (like PCA).

* **Preprocessing Tools**
  It helps clean and prepare your data — scale numbers, handle missing values, or convert categories to numbers.

* **Testing and Evaluation**
  Scikit-learn has tools for splitting data, doing cross-validation, and checking model performance.

* **Workflow with Pipelines**
  You can connect multiple steps into one flow — clean data, train model, and make predictions — all in one pipeline.


### Preprocessing Data

Before training a model, your data must be clean. Scikit-learn gives tools for that:

* **Scaling Data**

  * `StandardScaler`: standardizes data (mean = 0, std = 1)
  * `MinMaxScaler`: scales data between 0 and 1
  * `RobustScaler`: works well if you have outliers

* **Handling Categorical Features**

  * `OneHotEncoder`: turns categories into 0s and 1s
  * `OrdinalEncoder`: turns categories into numbers like 0, 1, 2…

* **Filling Missing Values**

  * `SimpleImputer`: fills missing data with mean, median, or a constant

* **More Features**

  * Create new features using `PolynomialFeatures`
  * Convert numbers into groups (bins) with `KBinsDiscretizer`

### Using Pipelines

Pipelines let you combine all the steps — preprocessing and modeling — into one object. This makes your code clean and ensures the same steps happen to both training and test data.

### Algorithms and Modules I Used

Scikit-learn supports many types of ML models and tools:

* **Supervised Learning**

  * Classification: logistic regression, SVM, decision trees, random forest
  * Regression: linear regression, ridge, lasso

* **Unsupervised Learning**

  * Clustering: KMeans, DBSCAN
  * Dimensionality reduction: PCA

* **Model Tuning**

  * `GridSearchCV` and `RandomizedSearchCV` help you find the best settings

* **Metrics**

  * Accuracy, precision, recall, F1-score, ROC-AUC, etc.


### Step-by-Step Workflow I Follow

1. **Load Data**
   Usually with pandas.

2. **Clean and Prepare**
   Scale, encode, or fill missing values using scikit-learn tools.

3. **Build Pipeline**
   Chain everything together.

4. **Train the Model**
   Use `.fit()` on training data.

5. **Test the Model**
   Use `.score()` or other metrics to see how good it is.

6. **Tune the Model**
   Try different settings with grid search.

7. **Make Predictions**
   Use `.predict()` to get results on new data.


## Feature Engineering

Feature engineering is an important part of machine learning. It means changing and creating data (called **features**) so that the model can learn better and make better predictions.

### What is a Feature?

A feature is just a piece of information in your data — like a person’s age, income, or product price. Feature engineering is about using or making the right features for your model.

### Why Feature Engineering Is Important

* **Better accuracy:** Good features help models work better.
* **Less overfitting:** Using only useful features makes models simpler and safer.
* **Fixes messy data:** Helps clean missing values, outliers, or text data.

### Main Types of Feature Engineering

1. **Improve Features:** Clean or fix current data (like filling missing values or scaling numbers).
2. **Create New Features:** Combine or change features (e.g., turning "date" into "day of the week").
3. **Select Important Features:** Keep only the most useful ones.
4. **Extract Info:** Use tools like PCA to reduce feature numbers.
5. **Let Models Learn Features:** Deep learning can find features by itself (like in images or audio).

### Common Techniques

* **Encoding:** Turn categories into numbers (like "Male" → 0, "Female" → 1).
* **Scaling:** Make all numbers similar in size.
* **Filling Missing Data:** Use mean, median, or a fixed value.
* **Binning:** Group numbers (e.g., age 0–18 = “young”, 19–35 = “adult”).
* **Combining Features:** Make a new one from two others.
* **Use Domain Knowledge:** Think about the real-world meaning of the data.
