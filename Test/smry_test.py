
# importing modules
import urllib.request 
from bs4 import BeautifulSoup
import requests
  
# # providing url
url = "http://www.bbc.co.uk/news/business-38650976"
  
# # # opening the url for reading
# # html = urllib.request.urlopen(url)
  
# # # parsing the html file
# # htmlParse = BeautifulSoup(html, 'html.parser')
  
# # # getting all the paragraphs
# # for para in htmlParse.find_all("p"):
# #     print(para.get_text())
# def get_contents(link):
#     html = requests.get(link)
#     htmlParse = BeautifulSoup(html.text, 'html.parser')
#     contents = ""
#     for para in htmlParse.find_all("p"):
#         print(contents + para.get_text())

# get_contents(url)

from bs4 import BeautifulSoup
import requests
import urllib
url = 'https://www.aljazeera.com/news/2017/2/20/cyprus-an-island-divided'
r = requests.get(url)
html = r.text
soup = BeautifulSoup(urllib.request.urlopen(url), "html.parser")
# links = soup.find_all('div', {'class': 'image'})
# if links:
#     print(links[0].find('img')['src'])
for items in soup.find('img'):
    print(items['src'])
    # print(links[0].find('img')['title'])

# from urllib import request
# url = "http://www.bbc.co.uk/news/election-us-2016-35791008"
# html = request.urlopen(url).read().decode('utf8')
# html[:60]

# from bs4 import BeautifulSoup
# soup = BeautifulSoup(html, 'html.parser')
# title = soup.find('title')

# print(title) # Prints the tag
# print(title.string) # Prints the tag string content