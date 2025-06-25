# SOC-midterm-review
## Python Basics

Learning Python has been a very useful experience for me. As a beginner, I started with no prior knowledge, but by following tutorials, reading online resources, and practicing regularly, I have gained a good understanding of the basics. Below is a summary of the key topics I have covered so far.

### Getting Started

I began by installing Python 3 in Google Colab and done all my coding on colab only.

### Python Syntax and Philosophy

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

