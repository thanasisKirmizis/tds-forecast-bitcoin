#%% Import modules

import datetime
import requests
import pandas as pd
from pytrends import dailydata
import snscrape.modules.twitter as sntwitter
import pageviewapi


#%% Define scraping functions

def get_google_trends_data(keyword, from_date, to_date):
    
    """
    Gets daily Google Trends for keyword.
    Dates like: 'YYYY-MM-DD'

    """
    
    from_year, from_month = datetime.date.fromisoformat(from_date).year, datetime.date.fromisoformat(from_date).month
    to_year, to_month = datetime.date.fromisoformat(to_date).year, datetime.date.fromisoformat(to_date).month

    data = dailydata.get_daily_data(keyword, from_year, from_month, to_year, to_month)
    
    return data[keyword]

def get_wikipedia_data(article, from_date, to_date):
    
    """
    Gets wikipedia article page views.
    Dates like: 'YYYYMMDD'

    """
    
    response =  pageviewapi.per_article('en.wikipedia.org', article, from_date, to_date,
                                        access = 'all-access', agent = 'all-agents', granularity = 'daily')
    
    data = [x['views'] for x in response['items']]
    dates = [x['timestamp'] for x in response['items']]
    
    return pd.DataFrame({'date': dates, 'views': data})

def get_reddit_data(subreddit, from_date, to_date):

    """
    Gets reddit data from the pushshift api.
    Dates like: 1609459199 (seconds since epoch)

    """

    base_url = "https://api.pushshift.io/reddit/search/submission/"
    
    counts_list = []
    dates_list = []
    
    # Day to seconds calculations
    after = from_date
    day_in_secs = 24 * 60 * 60
    next_day = after + day_in_secs
    
    daily_count = 0
    
    # Loop through all pages
    while after < to_date:
        
        # Parameters
        payload = {'data_type': "submission",
                   'subreddit': subreddit,
                   'sort': "asc",
                   'limit': 100,
                   'after': str(after),
                   'before': str(next_day)
                  } 
        
        # Get the posts in batches of 100
        request = requests.get(base_url, params = payload)
        data = request.json()
        daily_count = daily_count + len(data['data'])
        
        # Next batch will continue from when the last post ended
        after = data['data'][-1]['created_utc'] + 1 if len(data['data']) > 0 else next_day
        
        # End of day, go to next one
        if after >= next_day:
            print("This day had:", daily_count, ". Going to next day...")
            counts_list.append(daily_count)
            dates_list.append(datetime.date.fromtimestamp(next_day - day_in_secs))
            next_day = next_day + day_in_secs
            daily_count = 0
    
    return pd.DataFrame({'date': dates_list, 'posts': counts_list})

def get_twitter_data(keyword, from_date, to_date):
    
    """
    Gets twitter posts containing specific keyword.
    Dates like: 'YYYY-MM-DD'
    
    """
    # Creating list to append tweet data to
    counts_list = []
    dates_list = []
    
    days = pd.date_range(start = from_date, end = to_date)
    
    for i in range(len(days)-1):
        
        # Using TwitterSearchScraper to count daily tweets
        daily_count = 0
        for item in sntwitter.TwitterSearchScraper(keyword + ' since:' + str(days[i].date()) + ' until:' + str(days[i+1].date())).get_items():
            daily_count = daily_count + 1
        
        print("Day", str(days[i].date()), "had:", daily_count, ". Going to next day...")
        
        dates_list.append(days[i].date())
        counts_list.append(daily_count)
        
    return pd.DataFrame({'date': dates_list, 'tweets': counts_list})


#%% Scrape desired data (might take a while)

# Get Google data and save them
google = get_google_trends_data('Bitcoin', '2013-10-01', '2020-12-15')
google.to_csv('data/google_data.csv')

# Get Wikipedia data and save them
wikipedia = get_wikipedia_data('Bitcoin', '20131001', '20201215')
wikipedia.to_csv('data/wikipedia_data.csv')

# Get Reddit data and save them
reddit = get_reddit_data('Bitcoin', 1380585600, 1608076800)
reddit.to_csv('data/reddit_data.csv')

# Get Twitter data and save them
twitter = get_twitter_data('Bitcoin', '2013-10-01', '2020-12-15')
twitter.to_csv('data/twitter_data.csv')
