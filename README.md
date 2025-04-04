# Data Science Student Learning Portfolio

This repository showcases my work as an **M.S. Computer Science (Data Science/AI) student**, with ongoing and future projects focused on key data science techniques. It highlights my commitment to learning and applying data analysis, machine learning, deep learning, and other advanced methods to solve real-world problems.

**Expanding My Learning with Cybersecurity & Red Teaming Applications:**

Alongside my data science studies, I have completed the **AI Red Teamer Job Role Path** from **Hack The Box** and **Google** ([Course Link](https://academy.hackthebox.com/path/preview/ai-red-teamer)). This training builds expertise in **AI security, adversarial attacks, red teaming strategies, and model exploitation**, aligned with Google’s **Secure AI Framework (SAIF)**. By integrating **offensive security testing**, I aim to understand and mitigate vulnerabilities in AI-driven applications, complementing my broader AI and data science learning.

---
## Hack The Box: AI Red Teamer Path [Proof of Completion](https://academy.hackthebox.com/achievement/586157/path/418)

![image](https://github.com/user-attachments/assets/479b4e98-5b57-496c-9888-153df3197c95)

#### Pointers to other GitHub Repositories as part of the Hack The Box AI Red Teamer Path

- **[Red Teaming TTPs for AI](https://github.com/CommodoreAlex/Red-Teaming-AI)**  
  This repository explores **Red Teaming tactics, techniques, and procedures (TTPs)** applied to AI systems. It includes methods for assessing and exploiting vulnerabilities in AI models, focusing on adversarial attacks, data poisoning, and model manipulation. The goal is to understand how to test and secure AI models by simulating real-world attacks.

- **[Prompt Injection Attacks against AI Endpoints and Models](https://github.com/CommodoreAlex/AI-Prompt-Injection)**  
  This repository focuses on **prompt injection attacks**, which manipulate AI model inputs to cause unintended behavior. It demonstrates how prompt injection can exploit vulnerabilities in AI systems, emphasizing potential data leaks, system manipulation, and other malicious impacts. The repository also provides techniques to mitigate these risks and secure AI models.

----
### Jupyter Notebooks

**Spam Classification:**
- **Focus**: Classifying SMS messages as spam or ham using Naive Bayes Theorem to understand adversarial AI techniques.
- **Tools**: Python, scikit-learn, pandas, NumPy, Joblib, Jupyter Notebook
- Notebook: [Spam Classification](Spam.ipynb)

**Network Anomaly Detection:**  
- **Focus**: Detecting malicious network traffic using Random Forests to identify anomalies that may indicate intrusions, malware, or unauthorized access.  
- **Tools**: Python, scikit-learn, pandas, NumPy, Seaborn, Matplotlib, Joblib, Jupyter Notebook  
- Notebook: [Network Anomaly Detection](Network.ipynb)  

**Malware Classification & Analysis:**
- **Focus**: Classifying malware samples based on their **binary-to-image representations** using **deep learning techniques**. This approach enables **automated malware classification**, reducing reliance on manual reverse engineering and improving threat intelligence. By leveraging **Convolutional Neural Networks (CNNs)** and **transfer learning**, we can enhance detection accuracy and model robustness.
- **Tools**: Python, PyTorch, TensorFlow, scikit-learn, OpenCV, Pandas, NumPy, Seaborn, Matplotlib, Joblib, Jupyter Notebook
- Notebook: [HTB Academy Malware Classification](mal.ipynb)

**Hack The Box Skills Assessment: IMDB Movie Reviews:**
- **Focus**: Developing a machine learning model to classify movie reviews from the IMDB dataset as either positive (1) or negative (0) using natural language processing (NLP) techniques. The project involves training a sentiment classification model and evaluating its performance, with an emphasis on refining machine learning models for sentiment analysis tasks.
- **Tools**: Python, scikit-learn, NLTK, pandas, NumPy, Jupyter Notebook
- Notebook: [Hack The Box Skills Assessment](IMDB_HTB.ipynb)

**Red Teaming AI: Manipulating the Model:**  
- **Focus**: Exploring how machine learning models can be manipulated through adversarial techniques, including injection manipulation (ML01) and data poisoning (ML02). The project demonstrates how modifying input and training data affects model behavior, highlighting vulnerabilities in AI security.  
- **Tools**: Python, scikit-learn, NLTK, pandas, NumPy, Jupyter Notebook  
- Notebook: [Red Teaming AI: Manipulating the Model](Manipulate.ipynb)

**Red Teaming AI Skills Assessment: Data Poisoning**
- **Focus**: Understanding and implementing data poisoning techniques to manipulate the behavior of machine learning classifiers. The goal is to insert a backdoor into a spam classification system that misclassifies messages with a specific phrase as ham, while maintaining overall high accuracy.
- **Tools**: Python, scikit-learn, NLTK, pandas, NumPy, Jupyter Notebook  
- **Notebook**: [Data Poisoning Skills Assessment](RedSkillAssessment.ipynb)  

---
## Conventional Data Science Projects

These projects showcase my focused work in data science, machine learning, and data visualization.

**Exploratory Data Analysis (EDA) on Netflix Titles Dataset:**
- **Focus**: Data cleaning, trend analysis (genre, ratings, release year, certifications).
- **Tools**: Python, Pandas, Matplotlib, NumPy
- **Notebook**: [Netflix EDA Notebook](./netflix.ipynb)

**Expanding on Exploratory Data Analysis (EDA) on Bitcoin Historical Dataset:**
- **Focus**: Data cleaning, trend analysis of Bitcoin's price movements, volatility, and trading volume over time. This involves analyzing Bitcoin's opening, closing, high, and low prices, as well as the trading volume across different years (2014-2018), identifying patterns, and understanding market behavior.
- **Tools**: Python, Pandas, Plotly, NumPy, Dash
- **Notebook**: [Bitcoin Data Visualization Notebook](./Data.ipynb)

**Machine Learning Model for Predicting Telco Customer Churn:**
- **Objective**: Build a machine learning model to predict customer churn using the Telco Customer Churn dataset. The project involves data preprocessing, model creation, training, and evaluation to determine the likelihood of customer churn.
- **Approach**: Focused on testing multiple machine learning models, including Logistic Regression, Random Forest, and Gradient Boosting (XGBoost), to predict customer churn. The project involves data exploration, feature engineering, and model performance evaluation.
- **Tools & Libraries**: Python, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, XGBoost
- **Notebook**: [Telco Customer Churn Prediction Notebook](./ML.ipynb)

**Natural Language Processing with Hugging Face:**
- **Objective**: Explore Hugging Face’s NLP capabilities by utilizing transformer models for various text-processing tasks such as text classification, summarization, and question answering.
- **Approach**: Implement Hugging Face pipelines to simplify NLP workflows, experiment with tokenizers, and pre-trained models. The project involves loading and processing text, applying models, and evaluating outputs.
- **Tools & Libraries**: Python, Hugging Face Transformers, PyTorch
- **Notebook**: [Natural Language Processing with Hugging Face](Hugging.ipynb)

**Deep Learning: Simple MNIST Neural Network Model:**
- **Objective**: Explore the MNIST dataset and build a simple neural network to classify handwritten digits using deep learning techniques. Understand the basic principles of deep learning, including neural network architecture, activation functions, and optimization methods.
- **Approach**: Implement a multi-layer neural network with ReLU, sigmoid, and softmax activation functions. Train the model on the MNIST dataset and evaluate its accuracy using test data. Visualize misclassifications through a confusion matrix.
- **Tools & Libraries**: Python, TensorFlow, Keras, NumPy, Matplotlib, Seaborn
- **Notebook**: [Deep Learning: Simple MNIST Neural Network Model](MNIST1.ipynb)

**Reinforcement Learning: OpenAI Gym Practice:**
- **Objective**: Explore reinforcement learning using OpenAI Gym by implementing and training a deep Q-network (DQN) to solve the CartPole problem. Understand the principles of reinforcement learning, including the agent-environment interaction, reward maximization, and policy-based learning.
- **Approach**: Implement a DQN-based agent for the CartPole environment using Keras-RL. Train the agent to balance the pole by applying forces to the cart and optimizing the model's parameters with Q-learning techniques. Test and evaluate the agent’s performance across multiple episodes.
- **Tools & Libraries**: Python, OpenAI Gym, TensorFlow, Keras, Keras-RL, NumPy
- **Notebook**: [Reinforcement Learning: OpenAI Gym Practice](gym1.ipynb)

**Natural Language Processing with NLTK:**  
- **Objective**: Explore and implement core NLP techniques to process and analyze text data effectively, focusing on tokenization, POS tagging, named entity recognition (NER), and text normalization.  
- **Approach**: Utilize Python’s `NLTK` library to perform text preprocessing, including segmentation, stop word removal, stemming, and lemmatization. Apply POS tagging and NER to extract meaningful insights from text.  
- **Tools & Libraries**: Python, NLTK
- **Notebook**: [Natural Language Processing](NLP./ML.ipynb)

**FirstCuda: CUDA Programming in Jupyter Notebook for Beginners**
- **Focus**: Introduction to CUDA programming for high-performance computing tasks using NVIDIA GPUs within a Jupyter Notebook. It covers setting up CUDA with Jupyter/Anaconda, benchmarking CPU vs. GPU performance, and writing custom CUDA kernels for element-wise operations.
- **Tools**: Python, NumPy, CuPy, CUDA Toolkit
- **Notebook**: [CUDA Notebook](FirstCuda.ipynb)

**TensorFlow Churn Prediction: A First Experience**
- **Focus**: Building a simple neural network using TensorFlow to predict customer churn based on historical data. It covers data loading, model creation, training, and evaluation to predict customer churn.
- **Tools**: Python, TensorFlow, Keras, NumPy, Pandas
- **Notebook**: [TensorFlow Flow Churn Prediction Notebook](./Tensors.ipynb)

**Introduction to Pandas CRUD Operations:**
- **Focus**: This notebook covers the essential Create, Read, Update, and Delete (CRUD) operations in Pandas. It provides a hands-on introduction to handling data in Pandas, including loading datasets, manipulating data, and exporting the results.
- **Tools**: Python, Pandas
- **Notebook**: [Pandas CRUD Operations Notebook](Pandas.ipynb)

**CRUD Operations with NumPy:**
- **Focus**: This notebook explores the basic Create, Read, Update, and Delete (CRUD) operations using NumPy arrays, providing a foundation for data manipulation.
- **Tools**: Python, NumPy
- **Notebook**: [CRUD Operations with NumPy](./crud.ipynb)

---
## Skills & Tools

- **Languages**: Python
- **Libraries**: Pandas, Scikit-learn, TensorFlow, Keras, OpenAI Gym, Matplotlib
- **Machine Learning**: Supervised/Unsupervised Learning, Model Evaluation
- **Deep Learning**: CNNs, RNNs, DQN
- **Visualization**: Tableau, Plotly, Seaborn
- High-Performance Computing: CUDA, GPU Acceleration

---
## Video Resources for Learning Data Science

As I have been scouring the internet for many resources to learn theory, practical projects, and other related subject matter- I have mostly committed those to playlists. See below for resources:

* [Open AI Gym Playlist](https://www.youtube.com/playlist?list=PLhBFZf0L5I7oIFTNTclyvWRciXaVb76Yt)
* [MISC Data Science](https://www.youtube.com/playlist?list=PLhBFZf0L5I7qDhgcgKYuqsew0DgaCQJ27)
* [Data Science Lectures](https://www.youtube.com/playlist?list=PLhBFZf0L5I7qN_qb4P1lvrjhHtMd7403_)
* [CUDA](https://www.youtube.com/playlist?list=PLhBFZf0L5I7qK4syDgdElaY4K1QdZZGNh)
* [Malware Data Science](https://www.youtube.com/playlist?list=PLhBFZf0L5I7qGUHDPxZKgBfbgt8nRI0Oo)

----
