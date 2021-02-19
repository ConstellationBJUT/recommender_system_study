"""
@Time : 2021/2/19 1:47 PM 
@Author : Xiaoming
评价指标hit和mrr
"""
from tqdm import tqdm


def evaluate(df, total):
    hitrate_5 = 0
    mrr_5 = 0

    hitrate_10 = 0
    mrr_10 = 0

    hitrate_20 = 0
    mrr_20 = 0

    hitrate_40 = 0
    mrr_40 = 0

    hitrate_50 = 0
    mrr_50 = 0

    gg = df.groupby(['user_id'])

    for _, g in tqdm(gg):
        try:
            item_id = g[g['label'] == 1]['article_id'].values[0]
        except Exception as e:
            continue

        predictions = g['article_id'].values.tolist()

        rank = 0
        while predictions[rank] != item_id:
            rank += 1

        if rank < 5:
            mrr_5 += 1.0 / (rank + 1)
            hitrate_5 += 1

        if rank < 10:
            mrr_10 += 1.0 / (rank + 1)
            hitrate_10 += 1

        if rank < 20:
            mrr_20 += 1.0 / (rank + 1)
            hitrate_20 += 1

        if rank < 40:
            mrr_40 += 1.0 / (rank + 1)
            hitrate_40 += 1

        if rank < 50:
            mrr_50 += 1.0 / (rank + 1)
            hitrate_50 += 1

    hitrate_5 /= total
    mrr_5 /= total

    hitrate_10 /= total
    mrr_10 /= total

    hitrate_20 /= total
    mrr_20 /= total

    hitrate_40 /= total
    mrr_40 /= total

    hitrate_50 /= total
    mrr_50 /= total

    return hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50
