import implicit
import pandas as pd
import scipy
from implicit.als import AlternatingLeastSquares
import numpy as np
if __name__ == "__main__":
    data = pd.read_csv("datasets/user_artists.dat", sep='\s+')
    data.set_index(["userID", "artistID"], inplace=True)
    
    coo_array = scipy.sparse.coo_array((data.weight.astype(np.float32), (
        data.index.get_level_values(0),
        data.index.get_level_values(1),
    )))
    
    
    user_artist_arr = coo_array.tocsr()
    
    # from implicit.datasets.lastfm import get_lastfm

    # artists, users, artist_user_plays = get_lastfm()
    # print(artist_user_plays)
    from implicit.nearest_neighbours import bm25_weight

    # # weight the matrix, both to reduce impact of users that have played the same artist thousands of times
    # # and to reduce the weight given to popular items
    user_artist_arr = bm25_weight(user_artist_arr, K1=100, B=0.8)
    user_artist_arr = user_artist_arr.tocsr()
    # # get the transpose since the most of the functions in implicit expect (user, item) sparse matrices instead of (item, user)
    # user_plays = artist_user_plays.T.tocsr()
    # print(user_artist_arr)
    model = AlternatingLeastSquares()
    model.fit(user_artist_arr)
    user = data.index.get_level_values(1)[0]
    
    recommendations = model.recommend(user, user_artist_arr[user])
    
    for recommendation in recommendations:
        print(recommendation)