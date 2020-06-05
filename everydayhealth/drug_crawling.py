# -*- encoding: utf-8 -*-
import urllib.request
import pandas as pd
from bs4 import BeautifulSoup

def make_link(url, drug, page_count):
    return url + drug + '/reviews/' + str(page_count)

def get_comment(url, drug):
    label_name = drug + '_Review'
    title = '191115' + drug + '.xlsx'
    review_list = []
    label = [label_name]
    for page in range(1, 3):
        d_url = make_link(url, drug, page)
        req = urllib.request.Request(d_url, headers = {'User-Agent': 'Mozilla/5.0'})
        webpage = urllib.request.urlopen(req).read()
        soup = BeautifulSoup(webpage, 'html5lib')
        html = soup.find('div', {'id': 'reviews-target'})
        reviews = html.findAll('div', {'class': 'review-container row'})
        
        for review in reviews:
            review_text = review.find('p', {'itemprop': 'reviewBody'}).getText().strip()
            review_list.append(review_text)
        print(d_url)
        
        
    r_df = pd.DataFrame(review_list, columns = label)
    r_df.to_excel(title)


url = 'https://www.everydayhealth.com/drugs/'
drug = 'ibuprofen'
get_comment(url, drug)