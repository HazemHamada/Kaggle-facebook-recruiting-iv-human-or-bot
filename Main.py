import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder


# load data
data = pd.read_csv("train.csv")
bids = pd.read_csv("bids.csv")
bids = bids.sort_values('time')
test = pd.read_csv("test.csv")
sampleSubmission = pd.read_csv("sampleSubmission.csv")
########################################################################################################################

# data cleaning

bids = bids.dropna()
data = data.dropna()
data = data.drop_duplicates()
bids = bids.drop_duplicates()
encoder = LabelEncoder()
cat_features = ['merchandise', 'device', 'country']
encoded = bids[cat_features].apply(encoder.fit_transform)
bids = bids[['bid_id', 'bidder_id', 'auction', 'time', 'ip', 'url']].join(encoded)
########################################################################################################################

# feature generation

k1 = bids.groupby('bidder_id').nunique()
k2 = bids.groupby('bidder_id').apply(lambda g: g.groupby('country').nunique().mean())
k3 = bids.groupby('bidder_id').apply(lambda g: g.groupby('merchandise').nunique().mean())
x1 = bids.groupby('bidder_id').bid_id.count()
x2 = k1['auction']
x3 = k2.bid_id
x4 = k1['url']
x5 = k1['time']
x6 = k1['device']
x7 = bids.groupby('bidder_id').time.apply(lambda x: x.apply(lambda t: t-t % 100000000000).value_counts().max())
x8 = bids.groupby('bidder_id').time.apply(lambda x: x.apply(lambda t: t-t % 1000000000000).value_counts().max())
x9 = bids.groupby('bidder_id').time.apply(lambda x: x.apply(lambda t: t-t % 10000000000).value_counts().max())
x10 = bids.groupby('bidder_id').time.apply(lambda group: (group.sort_values().diff().fillna(group.mean())).median())
x11 = k1['ip']
x12 = k3.bid_id
x13 = bids.groupby('bidder_id').apply(lambda g: g.groupby('auction').time.median().mean())
x14 = k1['country']

times = bids.groupby('auction').time.min().reset_index()
times = times.rename(columns={'time': 'startt'})
times2 = bids.groupby('auction').time.max().reset_index()
times2 = times2.rename(columns={'time': 'endt'})
times = pd.merge(times, times2, on='auction', how='left')
times['duration'] = times.endt - times.startt
times['short'] = 1.0 * (times['duration'] < 3.01 * 4547368124071.8799)
bids = pd.merge(bids, times[['auction', 'short', 'startt', 'endt']], on='auction', how='left')
# time from bid until end of auction
bids['t_until_end'] = bids.endt - bids.time
bids['t_since_start'] = bids.time - bids.startt


x15 = bids.groupby('bidder_id')['short'].mean()
x15._set_name(name='x15')
x16 = bids.groupby('bidder_id')['t_until_end'].median()
x16._set_name(name='x16')
x = x16.rename(columns={'t_until_end': 't_until_end_median'})
x17 = bids.groupby('bidder_id')['t_since_start'].median()
x17._set_name(name='x17')


x_total = pd.merge(x1._set_name(name='x1'), x2._set_name(name='x2'),on='bidder_id',how='left')
x_total = pd.merge(x_total, x3._set_name(name='x3'),on='bidder_id',how='left')
x_total = pd.merge(x_total, x4._set_name(name='x4'),on='bidder_id',how='left')
x_total = pd.merge(x_total, x5._set_name(name='x5'),on='bidder_id',how='left')
x_total = pd.merge(x_total, x6._set_name(name='x6'),on='bidder_id',how='left')
x_total = pd.merge(x_total, x7._set_name(name='x7'),on='bidder_id',how='left')
x_total = pd.merge(x_total, x8._set_name(name='x8'),on='bidder_id',how='left')
x_total = pd.merge(x_total, x9._set_name(name='x9'),on='bidder_id',how='left')
x_total = pd.merge(x_total, x10._set_name(name='x10'),on='bidder_id',how='left')
x_total = pd.merge(x_total, x11._set_name(name='x11'),on='bidder_id',how='left')
x_total = pd.merge(x_total, x12._set_name(name='x12'),on='bidder_id',how='left')
x_total = pd.merge(x_total, x13._set_name(name='x13'),on='bidder_id',how='left')
x_total = pd.merge(x_total, x14._set_name(name='x14'),on='bidder_id',how='left')
x_total = pd.merge(x_total, x15._set_name(name='x15'),on='bidder_id',how='left')
x_total = pd.merge(x_total, x16._set_name(name='x16'),on='bidder_id',how='left')
x_total = pd.merge(x_total, x17._set_name(name='x17'),on='bidder_id',how='left')


def log_entropy(x):
    e = np.sum(np.log(np.array(range(1, np.sum(x)))))
    for i in x:
        e -= np.sum(np.log(np.array(range(1, i))))
    return e


a = bids[['bidder_id', 'auction', 'ip']].groupby(['bidder_id', 'auction', 'ip']).size().reset_index()
a = a.rename(columns={0: 'bids_per_auction_per_ip'})
b = a.groupby(['bidder_id', 'auction']).bids_per_auction_per_ip.apply(log_entropy).reset_index()
b = b.rename(columns={'bids_per_auction_per_ip': 'bids_per_auction_per_ip_entropy'})
c = b.groupby('bidder_id').bids_per_auction_per_ip_entropy.median().reset_index()
c = c.rename(columns={'bids_per_auction_per_ip_entropy': 'bids_per_auction_per_ip_entropy_median'})
X = pd.merge(x_total, c[['bidder_id', 'bids_per_auction_per_ip_entropy_median']], on='bidder_id', how='left')
c = b.groupby('bidder_id').bids_per_auction_per_ip_entropy.mean().reset_index()
c = c.rename(columns={'bids_per_auction_per_ip_entropy': 'bids_per_auction_per_ip_entropy_mean'})
X = pd.merge(X, c[['bidder_id', 'bids_per_auction_per_ip_entropy_mean']], on='bidder_id', how='left')
b = bids[['bidder_id', 'auction', 'ip']].groupby(['bidder_id', 'auction']).ip.nunique().reset_index()
b = b.rename(columns={'ip': 'ips_per_bidder_per_auction'})
c = b.groupby('bidder_id').ips_per_bidder_per_auction.median().reset_index()
c = c.rename(columns={'ips_per_bidder_per_auction': 'ips_per_bidder_per_auction_median'})
X = pd.merge(X, c[['bidder_id', 'ips_per_bidder_per_auction_median']], on='bidder_id', how='left')
c = b.groupby('bidder_id').ips_per_bidder_per_auction.mean().reset_index()
c = c.rename(columns={'ips_per_bidder_per_auction': 'ips_per_bidder_per_auction_mean'})
x_total = pd.merge(X, c[['bidder_id', 'ips_per_bidder_per_auction_mean']], on='bidder_id', how='left')

x_total = x_total.rename(columns={'bids_per_auction_per_ip_entropy_median': 'x18'})
x_total = x_total.rename(columns={'bids_per_auction_per_ip_entropy_mean': 'x19'})
x_total = x_total.rename(columns={'ips_per_bidder_per_auction_median': 'x20'})
x_total = x_total.rename(columns={'ips_per_bidder_per_auction_mean': 'x21'})
########################################################################################################################

# Preprocessing




########################################################################################################################

#data splitting
X = pd.merge(data[['bidder_id', 'payment_account', 'address']], x_total, how='inner', left_on=['bidder_id'], right_on=['bidder_id'])
Y = data[['outcome']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=1)
X_val_train, X_val_test, Y_val_train, Y_val_test = train_test_split(X_val, Y_val, test_size=0.2, random_state=1)






