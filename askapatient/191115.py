# -*- encoding: utf-8 -*-
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

def make_link(url, drugid, drug, page_count):
    return url + str(drugid) + '&name=' + drug + '&page=' + str(page_count)
url = 'https://www.askapatient.com/viewrating.asp?drug='
drugid = 20998
drug = 'CELEBREX'
page = 1
d_url = make_link(url, drugid, drug, page)

#DRIVER_DIR ='C:/Users/god12/Anaconda3/chromedriver'
#driver = webdriver.Chrome(DRIVER_DIR)
#driver.implicitly_wait(5)
#driver.get(url)
#ele = driver.find_element_by_class_name('tableBar').text
#print(ele)
profile = webdriver.FirefoxProfile()
profile.set_preference("network.proxy.type", 1)
profile.set_preference("network.proxy.socks", "127.0.0.1")
profile.set_preference("network.proxy.socks_port", 9050)
profile.update_preferences()
driver = webdriver.Firefox(profile)
driver.get('http://icanhazip.com/')
#driver.get('http://icanhazip.com/')
print(driver.page_source)
driver.quit()