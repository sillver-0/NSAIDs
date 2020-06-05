import urllib.request
import pandas as pd
from bs4 import BeautifulSoup

reviews = []
conditions = []
label = ['review']

def make_link(url, drug, page_count):
    return url + drug + '/?page=' + str(page_count)

def get_comment(url, drug_name):
    for page in range(1, 23):
        d_url = make_link(url, drug_name, page)
        req = urllib.request.Request(d_url, headers = {'User-Agent': 'Mozilla/5.0'})
        webpage = urllib.request.urlopen(req).read()
        html = BeautifulSoup(webpage, 'html5lib')
        comments = html.find_all('p', {'class':'ddc-comment-content'})
            
        for comment in comments:
            comment = comment.get_text()
            ind = comment.find(':')
            if 'For' in comment:
                conditions.append(comment[:ind])
                reviews.append(comment[ind+2:])
            else:
                cond_ind = comment.find('for')
                conditions.append(comment[cond_ind+4:ind])
                reviews.append(comment[ind+2:])            
        
        print(d_url)    
   
    r_df = pd.DataFrame(reviews, columns = label)
    r_df.loc[:, 'condition'] = pd.Series(conditions)
    r_df.to_excel('200120naproxen.xlsx')
        
        
d_url = 'https://www.drugs.com/comments/'
drug_t = 'naproxen'
get_comment(d_url, drug_t)