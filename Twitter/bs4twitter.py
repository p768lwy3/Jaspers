""" This is a small project trying not to use Twitter API to crawl the tweets from spec. account"""

import urllib3
import goslate
from bs4 import BeautifulSoup
import string
import operator

user_agent = {'user-agent': 'Mozilla/5.0 (Windows NT 6.3; rv:36.0) ..'}
HTML_PARSER = 'html.parser'

stop = set(('and', 'or', 'not', 'i', ' ', 'the', '', 'will', 'be', 'is', 'in', 'of', 'on', 'to', 'a', 'so', 'at'))

def get_tweet(_URL):
  """ Using urllib and BeautifulSoup to crawl the twitter from the URL directly... """
  gs = goslate.Goslate()
  http = urllib3.PoolManager(10, headers=user_agent)
  response = http.request('GET', _URL)
  soup = BeautifulSoup(response.data, HTML_PARSER)

  f = open('crawled_twitter.txt', 'wb')
  for tweet in soup.find('ol', attrs = {'class':'stream-items'}).find_all('li'):
    if tweet.find('p') is not None:
      tweet_find = tweet.find('p').text
		#if type(tweet_find) == str:
			#tweet_find = str(tweet_find, 'utf-8', errors = 'ignore')
		#else:
			#tweet_find = str(tweet_find)
    if isinstance(tweet_find, str):
      tweet_find = tweet_find.encode('utf-8')
    else:
      tweet_find = str(tweet_find).encode('utf-8')
    f.write(tweet_find + b'\n')
    #print(tweet_find)

def count(fname):
  """ Count the times of words occured... """
  wordcount = {}
  with open(fname, 'rb') as f:	
    for word in f.read().split():
      word = word.decode('utf-8')
      word = word.translate(str.maketrans('','',string.punctuation))
      word = word.lower()
      if word not in stop:
        if word not in wordcount:
          wordcount[word] = 1
        else:
          wordcount[word] += 1
    wordcount = sorted(wordcount.items(), key = operator.itemgetter(1), reverse=True) 
    for k,v in wordcount:
      print(k, v)

def main():
  #get_tweet('https://twitter.com/realdonaldtrump')
  #get_tweet('https://mobile.twitter.com/realDonaldTrump')
  count('crawled_twitter.txt')
	
if __name__ == '__main__':
  main()
