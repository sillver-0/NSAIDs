# -*- encoding: utf-8 -*-
import urllib.request
import pandas as pd
from bs4 import BeautifulSoup

def make_link(url, drug, page_count):
    return url + drug + '/reviews/' + str(page_count)

def get_comment(url, drug):
    label_name = drug + '_Review'
    title = drug + 'review1.xlsx'
    reviews = []
    label = [label_name]
    for page in range(1, 4):
        d_url = make_link(url, drug, page)
        req = urllib.request.Request(d_url, headers = {'User-Agent': 'Mozilla/5.0'})
        webpage = urllib.request.urlopen(req).read()
        soup = BeautifulSoup(webpage, 'html5lib')
        html = soup.find('ul', {'class': 'p-t-3 bg-grey-1 brd-grey-2 brd-t-1'})
        lis = html.findAll('li')
        for li in lis:
            review_text = li.findAll('span')
            if len(review_text[1].getText().strip()) <= 7:
                review = review_text[0].getText().strip()
            else:
                review = review_text[1].getText().strip()
            reviews.append(review)
        print(d_url)
    r_df = pd.DataFrame(reviews, columns = label)
    r_df.to_excel(title)
        
url = 'https://www.iodine.com/drug/'
drug = 'celebrex'
get_comment(url, drug)

#페이지변화없음