# packages for reddit webscraping
import praw
import pandas as pd
from praw.models import MoreComments

r = praw.Reddit(client_id = "9614FBzxS9q4N73AoN_rOg",
                client_secret = "m2GpCwbBWPxHe7J0JAZKeV914WmoWw",
                user_agent = "Prof453")

# set search parameters
q = '' # query
sub = 'premed' #subreddit - multiple subreddits can be combined with a + i.e. 'CryptoCurrency + SatoshiStreetBets'
sort = "top" #top, all, etc.
limit = 50

# create a search to the subreddit
top_posts = r.subreddit(sub).search(q, sort = sort, limit = limit)

# initialize empty list

total_posts = list()

# create loop to iterate over the top posts and create a dictionary to store the scraped data in order collected
# append the newly created dictionary to the empty list

for post in top_posts:
    # print(vars(post)) # print all properties
    Title = post.title,
    Score = post.score,
    Num_Comments = post.num_comments,
    Publish_Date = post.created,
    Link = post.permalink,
    data_set = {"Title": Title[0], "Score": Score[0],
                "Number of Comments": Num_Comments[0],
                "Publish Date": Publish_Date[0],
                "Link": 'https://www.reddit.com' + Link[0]}
    total_posts.append(data_set) # append to list

# create csv file with data
reddit_df = pd.DataFrame(total_posts)
reddit_df.to_csv('reddit_data.csv', sep = ',', index=False)

# get comments from a specific post
hot_posts = praw.reddit.subreddit(url)
submission = redd.submission(url = "https://www.reddit.com/r/premed/comments/wcy6xb/list_of_is_vs_oos_acceptances_for_do_schools/")
# or
#submission = reddit.submission(id = "")

# get top comments
submission.comments.replace_more(limit = 0)
for comment in submission.comments.list():
    print(comment.body)

