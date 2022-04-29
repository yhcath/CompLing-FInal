import os
import re
import tweepy
import datetime
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import string


TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")

class TwitterClient(object):
	'''
	Generic Twitter Class for sentiment analysis.
	'''
	def __init__(self):
		'''
		Class constructor or initialization method.
		'''

		# attempt authentication
		try:
			# create OAuthHandler object
			self.auth = tweepy.OAuth2BearerHandler(TWITTER_BEARER_TOKEN)

			# create tweepy API object to fetch tweets
			self.api = tweepy.API(self.auth)
		except:
			print("Error: Authentication Failed")

	def remove_noise(self, tweet, stop_words=()):
		tweet_tokens = tweet.split()

		cleaned_tokens = []

		for token, tag in pos_tag(tweet_tokens):
			token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
						   '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
			token = re.sub("(@[A-Za-z0-9_]+)", "", token)
			token = re.sub("[?.,:;]", "", token)

			if tag.startswith("NN"):
				pos = 'n'
			elif tag.startswith('VB'):
				pos = 'v'
			else:
				pos = 'a'

			#lemmatizer = WordNetLemmatizer()
			#token = lemmatizer.lemmatize(token, pos)

			if len(token) > 0 and token not in string.punctuation:
				cleaned_tokens.append(token.lower())
		return cleaned_tokens

	def get_timestamp(self, created_time):
		year = str(created_time.year)
		month = str(created_time.month)
		if len(month) < 2:
			month = f"0{month}"
		day = str(created_time.day)
		if len(day) < 2:
			day = f"0{day}"
		hour = str(created_time.hour)
		if len(hour) < 2:
			hour = f"0{hour}"
		minute = str(created_time.minute)
		if len(minute) < 2:
			minute = f"0{minute}"
		return f"{year}{month}{day}{hour}{minute}"
		#'202202060000'

	def get_tweets(self, query, start, end, count):
		'''
		Main function to fetch tweets and parse them.
		'''
		# empty list to store parsed tweets
		tweets = []

		try:
			recieved_count = 0
			fetched_tweets = []
			while recieved_count < count: #count
				#result = self.api.search_tweets(q=query, count=100)
				result = self.api.search_full_archive(label='development', query=query, fromDate=start, toDate=end, maxResults=100)
				batch_count = 0
				for tweet in result:
					if tweet.author.verified:
						continue
					if re.match(r'^RT @', tweet.text):
						continue
					if batch_count + recieved_count >= count:
						continue
					fetched_tweets.append(tweet)
					batch_count += 1
				recieved_count += batch_count
				earliest_created_at = result[-1].created_at - datetime.timedelta(minutes=1)
				end = self.get_timestamp(earliest_created_at)
			# call twitter api to fetch tweets
			# fetched_tweets = self.api.search_full_archive(label='development', query=query, fromDate=start, toDate=end, maxResults=count)

			# parsing tweets one by one
			for tweet in fetched_tweets:
				# empty dictionary to store required params of a tweet
				parsed_tweet = {}
				parsed_tweet['id'] = tweet.id_str
				# saving text of tweet
				try:
					parsed_tweet['text'] = tweet.extended_tweet['full_text']
				except AttributeError:
					if tweet.truncated:
						continue
					parsed_tweet['text'] = tweet.text

				# appending parsed tweet to tweets list
				if tweet.retweet_count > 0:
					# if tweet has retweets, ensure that it is appended only once
					if parsed_tweet not in tweets:
						tweets.append(parsed_tweet)
				else:
					tweets.append(parsed_tweet)

			# return parsed tweets
			return tweets

		except tweepy.TwitterServerError as e:
			# print error (if any)
			print("Error : " + str(e))

def main():
	# creating object of TwitterClient Class
	api = TwitterClient()
	# calling function to get tweets
	query = '"#eileengu" lang:en'
	start = '202202060000'
	end = '202202220000'
	count = 600
	tweets = api.get_tweets(query, start, end, count)

	stop_words = stopwords.words('english')

	result_csv = "id\ttext\n"
	for i, tweet in enumerate(tweets):
		id = tweet['id']
		text = " ".join(tweet['text'].split())
		result_csv += f"{id}\t{text}\n"

	with open('sentiment_nonrandomized.tsv', 'w') as csv_file:
		csv_file.write(result_csv)

if __name__ == "__main__":
	# calling main function
	main()

	print("Done")
