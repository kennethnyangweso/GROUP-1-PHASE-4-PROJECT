# AMAZON CUSTOMER REVIEW
## 1.0 BUSINESS UNDERSTANDING
Amazon is a global e-commerce and technology company, founded by Jeff Bezos in 1994. Initially focused on online book sales, it has since expanded into various sectors including retail, cloud computing (AWS), artificial intelligence, streaming (Amazon Prime Video), and logistics. Amazon is known for its customer-centric approach, innovative technologies, and vast product selection. It operates worldwide, with its core business encompassing e-commerce, Amazon Web Services (AWS), and devices such as Kindle and Alexa. Amazon also has a significant presence in the entertainment, electronics, Accessory, Smart home and healthcare industries. The company continues to drive advancements in automation, cloud services, and digital infrastructure.

Our focus with the amazon market in e-commerce is the customer reviews. In the online market of today, customer reviews were an essential part of purchasing decisions. Amazon, being a giant online store, collects millions of product reviews that indicate customer satisfaction, product quality, and overall user experience. It is not efficient, however, to process such vast data manually and is time-consuming.

Sentiment analysis enables companies to analyze customers' feedback automatically, extract meaningful information, and make knowledgeable input.
# 1.1 PROBLEM STATEMENT
Amazon gets millions of reviews; it faces the challenge of processing vast amounts of text data efficiently. Manually reading and analyzing each review is impractical. This is where automated sentiment analysis comes into play. By leveraging Natural Language Processing (NLP) and machine learning algorithms, Amazon can categorize reviews as positive, negative, or neutral. This automated system can quickly sift through large volumes of customer feedback, providing businesses with insights into customer satisfaction, product performance, and emerging trends. Sentiment analysis not only helps businesses understand the emotional tone of reviews but also aids in identifying key themes and areas for improvement, ultimately enhancing the customer experience and driving better decision-making.
## 1.2 OBJECTIVES
## 1.2.1 Main Objective
To accurately determine the overall emotional tone (positive, negative, or neutral) of customer reviews.
## 1.2.2 Specific Objectives
* Identify trends in customer satisfaction.

* Improve customer experience by addressing negative feedback.

* Help businesses optimize their product offerings based on user sentiment.
## 1.3 Key Business Questions
* What percentage of customer reviews are positive, negative, or neutral?
* Are there specific features or keywords associated with  reviews?
* Can sentiment analysis help predict potential or customer dissatisfaction?
* Can the  business use sentiment insights to improve product quality and customer support?
## 1.4 Why Now?
1. **Increasing reliance on online reviews:**

- Consumers make purchasing decisions based on online reviews more than ever before. Analyzing sentiment helps businesses understand customer feedback in real-time.

2. **Competitive business landscape:**

- Companies must react quickly to customer sentiments to stay ahead of competitors. Delayed sentiment analysis could mean losing customers to competitors with better service.

3. **Managing brand reputation in real time:**

- Social media and review platforms spread negative feedback instantly. Businesses need fast, automated sentiment analysis to detect issues and address them proactively.

4. **Data-driven decision-making:**

- Instead of relying on guesswork, businesses can use sentiment analysis now to guide product improvements, customer support strategies, and marketing campaigns.

5. **Rise of AI and automation:**

- With advancements in AI models such as LSTM, sentiment analysis is more accurate than ever. Now is the best time to implement these models for business intelligence.
## 1.5 Metric of Success
1. **Technical Success Metrics**

These metrics assess the performance of your machine learning models in terms of accuracy, reliability, and scalability.

**Model Performance Metrics**

 **Accuracy:** Measures the overall correctness of the model’s predictions. Higher accuracy indicates better classification. We aim to achieve an accuracy of 85% and above

**Precision, Recall, and F1-score:**

- Precision: Measures how many predicted positive sentiments are actually positive.

- Recall (Sensitivity): Measures how well the model identifies all positive sentiments.

- F1-score: Harmonic mean of precision and recall; balances false positives and false negatives.

**Confusion Matrix:** Helps analyze misclassification between positive, neutral, and negative reviews.

**Loss Function (Cross-Entropy Loss for Classification):** Lower loss values indicate better model predictions.

**Scalability:** Ensures the model can handle large datasets efficiently in real-world deployment.
2. **Business Success Metrics**

These metrics determine the impact of the sentiment analysis system on business outcomes.

**Customer Experience & Retention**

- Increase in Customer Satisfaction: A rise in positive sentiment over time indicates improved customer experience.

- Decrease in Negative Reviews: A reduction in complaints suggests that business improvements align with customer expectations.

- Improved Response Time: Faster identification of negative reviews enables quicker customer service interventions.

**Revenue & Sales Impact**

- Higher Product Ratings & Sales Growth: Positive sentiment correlates with increased sales and product trust.

- Reduction in Return Rates: Fewer negative reviews and complaints may indicate improved product quality, reducing return rates.

- Customer Retention Rate: If sentiment analysis helps address concerns proactively, retention should increase.

**Operational Efficiency**

- Automation of Review Analysis: Reduces manual effort in sorting and responding to reviews.

- Enhanced Decision-Making: Insights from sentiment trends guide product improvements and marketing strategies.

# 2.0 DATA UNDERSTANDING
The dataset used for this sentiment analysis project consists of Amazon product reviews, which provide insights into customer opinions about various products. It contains 1,597 records with 27 columns, capturing details about the product, review content and user feedback.
# 3.0 DATA PREPARATION
This phase involves transforming raw data into a structured and clean format suitable for modeling. This step ensures that the dataset is free from inconsistencies, missing values, and unnecessary variables while preparing it for machine learning and deep learning algorithms.
## 3.1 Data cleaning
This is the process of cleaning the dataset by:

* Dropping unnecessary columns
* Checking and dealing with missing values
* Dropping the duplicates
* Changing the columns format
* Checking for outliers
## 3.2 Feature Engineering
Feature engineering is crucial for improving model accuracy and interpretability also assists in exploratory data analysis. In this project, the goal is to extract meaningful insights from the available features and create new ones that enhance the model's predictive power.

- The feature engineering done for this project are:
- Converting date format to MM-YY-DD
- Mapping categories column to three distinct categories
- Distributing the reviews.rating column into three distinct categories
## 3.3 Text Preprocessing for NLP analysis
Text preprocessing in NLP involves several steps to clean and transform raw text into a format suitable for analysis. The key steps include:

1. Lowercasing – Convert all text to lowercase to ensure uniformity.

2. Removing Punctuation & Special Characters – Eliminates unnecessary symbols that do not contribute to meaning.

3. Tokenization – Splitting text into individual words or sentences.

4. Stopword Removal – Removing common words (e.g., "the," "is") that don’t add much value.

5. Lemmatization  – Converting words to their base or root form (e.g., "running" → "run").

## 3.4 Exploratory Data Analysis (EDA)
For this phase we will make visualizations that will help evaluate and analyze our business objectives and answer our business questions.

Visualizations crucial for this particular EDA include
- Histplot
- Countplot
- Donutplot
- Piechart
- Stacked barplot
- Barplot
- Lineplot

Steps to follow include:
- Univariate Analysis
- Bi-variate Analysis
- Multivariate Analysis

### 3.4.1 Univariate Analysis
![image](https://github.com/user-attachments/assets/f74595a3-497e-4c02-be16-16aee0ac1b5c)
##### Observations :

-  **Right-Skewed Distribution**: The majority of reviews have higher ratings (4 and 5). This indicates that most customers had a positive experience.

- **Fewer Low Ratings**: Very few reviews fall in the 1 to 3 rating range. This suggests that negative feedback is rare

### 3.4.2 Bi-variate Analysis
![image](https://github.com/user-attachments/assets/b26fbb0e-0b0f-47a0-a009-4fb7f772add0)
##### Observations

Electronics have the highest average rating, suggesting strong customer satisfaction, likely due to product quality and performance.

Accessories receive a moderate rating, indicating mixed feedback, possibly due to variability in product utility or durability.

Smart Home products have the lowest rating, implying customer concerns, which could be related to usability, compatibility, or reliability issues.

### 3.4.3 Multi-variate Analysis
![image](https://github.com/user-attachments/assets/86c27bc2-f31c-46d6-a02b-907b50e27175)
##### Observations
Amazon Dominates in Ratings: Amazon consistently receives higher average review ratings than Moshi across all three categories (Electronics, Accessories, Smart Home).

Electronics Have Highest Ratings: Both Amazon and Moshi have their highest average review ratings in the Electronics category.

Accessories Have Lowest Ratings: Accessories show the lowest average review ratings for both brands, indicating potential challenges or customer dissatisfaction in this category.

Moshi's Performance Varies: Moshi's performance is more varied across categories, with a noticeable drop in ratings for Accessories and Smart Home compared to Electronics.

# 4.0 MODELING
We prepared our data by:

-	Split the dataset into X (target variable) and y (independent variable).
-	Label encoded y in a format the models can process.
-	Used TF-IDF (Text Frequency- Inverse Document Frequency) for text. This technique is used to reflect the importance of a term in a document.
-	Handled class imbalance using Class weight.
-	Used pipelines in order to automate workflow for building a machine model.

The best performing models were:
	Support Vector machine (SVM).

SVM is a supervised learning algorithm used for classification and regression tasks. It is particularly powerful in high-dimensional spaces and works well when the data is not linearly separable.

The model performance before Hyperparameter Tuning:

•	Accuracy-  87.59%
•	Precision- 88%
•	Recall- 100%
•	F1 score- 93%

Here it is evident that the model was performing well with positive reviews but struggles with neutral and negative reviews. In short, the model was biased towards predicting positive reviews.

We hyperparameter tuned the model and used GridSearch CV to choose the best params for the model to perform better.

The model performance after Hyperparameter Tuning (the process of selecting the best hyperparameters for a machine learning model to optimize its performance ):
•	Accuracy-  88.97%
•	Precision- 92%
•	Recall- 96%
•	F1 score- 94%

The results showed that the model performed better since the accuracy increased by 1.38 and the biasness towards the positive review also reduced hence showed that the model could slightly distinguish between the neutral and negative reviews.

	Recurrent Neural Network with Long-Term-Short Memory (LSTM).
Recurrent Neural Network (RNN) is a type of artificial neural network designed for processing sequential data.

The model performance was as follows:
	Accuracy Base LSTM-GRU: 87%

Balanced RNN: 89% The balanced RNN model slightly outperforms the base model, showing an improvement of 2% in accuracy. However, accuracy alone is not a sufficient measure, so let's examine class-wise performance.

	Negative Sentiment Base LSTM-GRU: Precision: 40% | Recall: 57% | F1-score: 47%

Balanced RNN: Precision: 43% | Recall: 43% | F1-score: 43% Observation: The balanced model has slightly better precision but lower recall than the base model. While the F1-score is slightly lower, the model is now less prone to false positives in the negative class.

	Neutral Sentiment Base LSTM-GRU: Precision: 67% | Recall: 17% | F1-score: 27%
Balanced RNN: Precision: 80% | Recall: 33% | F1-score: 47% Biggest Improvement! The balanced RNN greatly improves recall for neutral sentiment (from 17% to 33%), meaning it identifies more neutral samples correctly. The precision also increases from 67% to 80%, making it much more reliable for predicting neutral sentiment.

	Positive Sentiment (Majority Class) Base LSTM-GRU: Precision: 91% | Recall: 95% | F1-score: 93%
Balanced RNN: Precision: 92% | Recall: 97% | F1-score: 94% Slight Improvement: The model retains high performance in positive sentiment detection, with a small increase in recall and F1-score.

	Macro & Weighted Averages Macro Average F1-score:
Base LSTM-GRU: 56%

Balanced RNN: 61% Improvement: The macro average F1-score increases, meaning the model is more balanced across all classes.
Weighted Average F1-score:

Base LSTM-GRU: 85%
RNN: 88% Observation: The overall model performance improves due to better recognition of neutral and negative sentiments.

![image](https://github.com/user-attachments/assets/261f4551-000e-435a-9a18-8bc48e4f6081)
 ##### Confusion Matrix Analysis
**Logistic Regression:**

True Positives (Positive Sentiment): 122 correctly classified as positive.

False Negatives (Positive misclassified): 3 classified as negative, 2 as neutral.

False Positives (Incorrect Positives): 5 negatives and 5 neutrals incorrectly classified as positive.

Neutral Class Weakness: Only 2 neutrals correctly classified; 1 misclassified as negative and 5 as positive.

**SVM:**

True Positives (Positive Sentiment): 122 correctly classified.

False Negatives (Positive misclassified): 2 classified as negative, 3 as neutral.

False Positives (Incorrect Positives): 5 negatives and 5 neutrals misclassified as positive.

Neutral Class Weakness: Only 3 neutrals correctly classified; 3 misclassified as positive.

![image](https://github.com/user-attachments/assets/8698c47e-b9ac-48bd-bb77-8fe2d29cedc3)

##### Confusion Matrix Analysis:
1. **Negative Class (Actual: Negative)**

- 6 correctly predicted as Negative.

- 1 misclassified as Positive.

- 0 misclassified as Neutral.

The model performs fairly well but misclassifies one sample as Positive.

2. **Neutral Class (Actual: Neutral)**

- 3 correctly classified as Neutral.

- 3 misclassified as Negative.

- 6 misclassified as Positive.

The Neutral class has poor performance, showing high misclassification rates. The model struggles to distinguish Neutral sentiments, frequently confusing them with Positive and Negative.

3. **Positive Class (Actual: Positive)**

- 119 correctly classified as Positive.

- 7 misclassified as Negative.

- 0 misclassified as Neutral.

The model does well in identifying Positive reviews, but there is some confusion with Negative sentiments.
# 5.0 MODEL DEPLOYMENT
## 5.1 Saving the model for deployment
Here we will save the model first using keras before we begin the deployment
## 5.2 Deploying the model using Streamlit
**Steps:**
1. **Install Required Dependencies**

- Ensure all necessary libraries, including Streamlit, TensorFlow, Pandas, NumPy, and Scikit-learn, are installed. These libraries are essential for running the model and displaying results on Streamlit.

2. **Load the Trained Model and Tokenizer**

- Since the model was already trained and saved as best_rnn_model.keras, it needs to be loaded into the app. Additionally, the tokenizer (tokenizer.pkl) is required for text preprocessing before making predictions.

3. **Create the Streamlit App (app.py)**

- A Python script (app.py) should be created to:

- Load the trained sentiment analysis model and tokenizer.**

- Accept user input for sentiment classification.

- Preprocess the text input before feeding it into the model.

- Display the predicted sentiment along with confidence scores or other relevant insights.

4. **Run the Streamlit App Locally**

- This is to test if the app is working correctly, run it locally. This will launch a web interface where you can input text and check the sentiment predictions.

5. **Prepare for Deployment**

- Before deployment, ensure that all project files, including the main script (app.py), trained model, tokenizer, and a list of required dependencies (requirements.txt), are in place. The requirements.txt file contains all the libraries needed for deployment.

6. **Deploy on Streamlit Cloud**

- Push the project files to a GitHub repository.

- Log in to Streamlit Cloud and create a new app.

- Connect the GitHub repository and select app.py as the main file.

- Click the deploy button to launch the application.

- Once deployed, the app will be accessible via a unique Streamlit Cloud URL.

# 6.0 CONCLUSIONS
- Most review ratings are 5 stars, indicating high customer satisfaction. For the ratings category, majority are "Excellent," with minimal negative reviews. Electronics dominate the product category while Accessories and Smart Home have fewer reviews. In Brands Amazon is more popular while Moshi has the least presence. Over 90% of reviews are positive; negative sentiment is minimal. Top Words in for Positive reviews highlight "Amazon" and "Kindle"; negative reviews mention "fire" and "tablet."

- Customer ratings vary significantly across product categories, with Electronics receiving the highest average ratings, while Accessories and Smart Home products are rated lower. This suggests that customers generally find electronics more satisfactory compared to other categories.

- Brand-wise, Amazon products receive notably higher ratings than Moshi, indicating stronger customer satisfaction and potentially better product quality, reliability, or brand trust associated with Amazon.

- Sentiment analysis of review text and titles reveals a strong correlation between sentiment and ratings. Positive reviews are associated with the highest average ratings, neutral reviews receive moderately high ratings, and negative reviews have the lowest. This pattern confirms that customer sentiment in text and titles aligns closely with their given star ratings.

- Regarding rating distribution across product categories, Electronics has the highest number of reviews, with a majority classified as "Excellent." However, there are still some reviews indicating bad or average quality, showing room for improvement. Accessories and Smart Home categories have significantly fewer reviews, with a more even distribution among rating categories, suggesting lower customer engagement or mixed feedback in these segments.

- The average review rating has fluctuated significantly over the years, showing no clear upward or downward trend.  While there are periods of higher ratings, there are also noticeable dips, indicating variability in customer satisfaction over time. This suggests potential factors influencing ratings that warrant further investigation, such as product changes, service adjustments, or shifts in customer expectations.

- There is a clear dominance of positive sentiment in reviews over the years, with a notable spike in positive reviews around 2023. Negative and neutral sentiments remain consistently low, suggesting overall customer satisfaction. The significant increase in positive reviews in recent years warrants further investigation to understand the contributing factors

- Finally, there is a clear dominance of the "Electronics" category in terms of count over the years, with a significant spike around 2023. "Accessories" show a low but consistent presence, while "Smart Home" has only a few occurrences towards the end of the timeframe. The dramatic increase in "Electronics" in 2023 warrants further investigation to understand the contributing factors to this surge.

- The best model was a deep learning model the Balanced Recurrent Neural Network Model which was saved and deployed for future use.It achieved an accuracy of above 85% with balanced precision and recall tradeoff. This was caused by the high class imbalance. The dataset contained more positive sentiments than even neutral and negative ones. After numerous attempts the imbalanced was somehow balanced though not perfectly.

# 7.0 RECOMMENDATIONS:

1. **Leverage High Customer Satisfaction for Brand Growth**
- Since most review ratings are 5 stars and the majority of reviews fall under the "Excellent" category, the company should highlight these positive reviews in marketing campaigns. Testimonials and ratings can be showcased to reinforce trust and attract new customers.

2. **Enhance Customer Experience for Lower-Rated Categories**
- Electronics receive the highest ratings, while Accessories and Smart Home products have lower ratings. The company should investigate customer concerns in these categories by analyzing negative reviews and improving product quality, features, or pricing strategies.

3. **Improve Moshi’s Brand Presence and Perception**
- Amazon dominates brand ratings, while Moshi has a weaker presence. To increase Moshi’s competitiveness, the company should invest in better product differentiation, marketing efforts, and customer engagement strategies. Offering discounts or bundling Moshi products with higher-rated brands may also improve perception.

4. **Monitor and Maintain High Sentiment Trends**
- With over 90% of reviews being positive and a surge in positive reviews around 2023, the company should analyze the factors contributing to this growth. If it is due to product improvements, customer service enhancements, or new product releases, these strategies should be reinforced to sustain the positive trend.

5. **Investigate Rating Fluctuations Over Time**
- The variability in average ratings over the years suggests changing customer expectations, potential product updates, or service variations. The company should analyze review trends alongside product changes or competitor activity to identify improvement opportunities.

6. **Capitalize on the Electronics Market Growth**
- The Electronics category has seen a massive spike in 2023. The company should explore this surge, possibly expanding its electronics product line, increasing inventory, or launching new features to maintain and capture more market share.

7. **Continue Leveraging Deep Learning for Sentiment Analysis**
- Since the Balanced Recurrent Neural Network Model performed best and was deployed, the company should continue using deep learning techniques for sentiment analysis. Regular model updates and fine-tuning based on new customer feedback will ensure continued accuracy and value from the deployed model.

  # THE END
