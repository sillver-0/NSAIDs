# -*- encoding: utf-8 -*-
import urllib.request
import pandas as pd
from bs4 import BeautifulSoup

def make_link(url, drugid, drug, page):
    drugid = str(drugid)
    page = str(page)
    return url + drugid + '-'+ drug + '+oral.aspx?drugid='+ drugid + '&drugname=' + drug + '+oral&pageIndex=' + page + '&sortby=3&conditionFilter=-500'

def get_comment(url, drugid, drug):
    label_name = drug + '_Review'
    title = '191115' + drug + '.xlsx'
    reviews = []
    label = [label_name]
    for page in range(0, 24):
        d_url = make_link(url, drugid, drug, page)
        req = urllib.request.Request(d_url, headers = {'User-Agent': 'Mozilla/5.0'})
        webpage = urllib.request.urlopen(req).read()
        soup = BeautifulSoup(webpage, 'html5lib')
        review_texts = soup.findAll('div', {'id': 'ratings_fmt'})
        for review in review_texts:
            for i in range(1, 6):
                id = 'comFull' + str(i)
                review_text = review.find('p', {'id': id}).getText()
                reviews.append(review_text)
        print(page)
    r_df = pd.DataFrame(reviews, columns = label)
    r_df.to_excel(title)

url = 'https://www.webmd.com/drugs/drugreview-'
drugid = 5166
drug = 'ibuprofen-oral'
get_comment(url, drugid, drug)