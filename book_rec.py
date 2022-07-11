import pipes
from telnetlib import XAUTH
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import pickle
from sklearn.neighbors import NearestNeighbors 
from scipy.sparse import csr_matrix
import csv

book = pd.read_csv("D:\Downloads\BX-CSV-Dump\BX-Books.csv", sep=';', on_bad_lines='skip', encoding="latin-1",low_memory=False)
user = pd.read_csv("D:\Downloads\BX-CSV-Dump\BX-Users.csv", sep=';', on_bad_lines= 'skip', encoding="latin-1")
rating = pd.read_csv("D:\Downloads\BX-CSV-Dump\BX-Book-Ratings.csv", sep=';', on_bad_lines= 'skip', encoding="latin-1")

book.columns = ['ISBN', 'BookTitle', 'BookAuthor', 'YearOfPublication', 'Publisher', 'ImageUrlS', 'ImageUrlM', 'ImageUrlL']
user.columns = ['UserID', 'Location', 'Age']
rating.columns = ['UserID', 'ISBN', 'BookRating'] 

c1_usid = rating['UserID'].value_counts()

rating = rating[rating['UserID'].isin(c1_usid[c1_usid >= 10].index)]

c2_usid = rating['BookRating'].value_counts()

rating = rating[rating['BookRating'].isin(c2_usid[c2_usid >= 10].index)]
c1 = rating['BookRating'].value_counts()

book_rating = pd.merge(rating, book, on='ISBN')

columns = ['BookAuthor', 'YearOfPublication', 'Publisher', 'ImageUrlS', 'ImageUrlM', 'ImageUrlL']
book_rating = book_rating.drop(columns, axis=1)
book_rating = book_rating.dropna()

bkrating_cnt = (book_rating.groupby(by = ['BookTitle'])['BookRating'].count().reset_index().rename(
columns = {'BookRating': 'TotalRating'})[['BookTitle', 'TotalRating']])

rating_bkrating_cnt = book_rating.merge(bkrating_cnt, left_on = 'BookTitle', right_on = 'BookTitle', how = 'left')

a = rating_bkrating_cnt['TotalRating']
b=rating_bkrating_cnt[rating_bkrating_cnt['TotalRating']>=10]

user['Age'] = user['Age'].fillna(user['Age'].mode()[0])
user.drop('Age',axis=1,inplace=True)
user.drop('Location',axis=1,inplace=True)

final = b.merge(user,left_on = 'UserID', right_on = 'UserID', how = 'left')
final = final.drop_duplicates(['UserID', 'BookTitle'])

final_pivot = final.pivot(index = 'BookTitle', columns = 'UserID', values = 'BookRating').fillna(0)
final_matrix = csr_matrix(final_pivot.values)

model = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model.fit(final_matrix)

finalmodel = 'Final_Model.sav'
pickle.dump(model, open(finalmodel, 'wb'))

def Final(name):
    finalmodel = "D:\Downloads\Final_Model.sav"
    ml = pickle.load(open(finalmodel, 'rb'))
    indices = ml.kneighbors(final_pivot.loc[[name]], 10, return_distance=False)
    print("Recommended Books:")
    x = []
    for index, value in enumerate(final_pivot.index[indices][0]):
        print((index+1),") ",value)
        x.append(value)
    return x    


def main():       
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Book Recommendation System</h1> 
    </div> 
    """  
    st.markdown(html_temp, unsafe_allow_html = True)      
    name = st.text_input("Enter the name of the book: ").strip()
    result = [] 
     
    if st.button("Recommended books :"): 
        result = Final(name) 
        print(result)
        st.dataframe(result)
if __name__=='__main__': 
    main()

