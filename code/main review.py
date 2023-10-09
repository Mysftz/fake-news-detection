import numpy as np, pandas as pd, csv
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
import os
dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))

df = pd.read_csv(dir+'/source/news.csv')
df.shape; labels = df.label

loop_count = 0
test_size_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
random_state_range = list(range(1, 20))
max_df_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
max_iter_range= list(range(1, 50))
total_loop_count = (len(test_size_range)*len(random_state_range)*len(max_df_range)*len(max_iter_range))
f = open(dir+'/results/Results.csv', 'w+')

for a in test_size_range:
    for b in random_state_range:
        for c in max_df_range:
            for d in max_iter_range:
                x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size = a, random_state = b)

                tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=c)
                tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
                tfidf_test=tfidf_vectorizer.transform(x_test)
                
                pac = PassiveAggressiveClassifier(max_iter=d)
                pac.fit(tfidf_train,y_train)
                y_pred = pac.predict(tfidf_test)

                accuracy = round((accuracy_score(y_test,y_pred))*100, 2)
                
                loop_count = loop_count+1
                loop_per = round((loop_count/total_loop_count)*100, 3)
                print(f'Loop: {loop_count}/{total_loop_count} ({loop_per}%) Completed', end='\r')
                
                results_list = [accuracy, a, b, c, d]
                with open(dir+'/results/Results.csv', 'a') as dataf:
                    write = csv.writer(dataf); write.writerows([results_list])
f.close()