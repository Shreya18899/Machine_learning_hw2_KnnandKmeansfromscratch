# Part 4 - Collaborative Filter

**Step 1: User interaction matrix**
Given that this is a movie-recommendation case, the construction of an user-interaction matrix is essential to develop a recommendation system. The user-interaction matrix would consist of the users as the row and the movie names as the column. The ratings provided by the user with respect to the particular movie would be used as the data. For each interaction record, assign the corresponding value to the respective cell in the matrix. For example, if User1 rated Movie1 with a score of 4, you would place a 4 in the cell at the intersection of User1 and Movie1. 


RK ==> I would change this
If users rated a movie on a scale of 1-4, we might say they loved the movie if they rated it a 4, liked it if they rated it a 3, were indifferent if they rated it a 2, and didn't like it if they rated it a 1. Instead of using these values, we would map them to represent the directionality. An example mapping might look like the following:
- 4 -> 10
- 3 -> 5
- 2 -> 0
- 1 -> -5


**Step 2: Distance Metric**
**new version 1**
For this problem, we want to use cosine similarity as our metric. This is for two reasons. One, we care about similarity (directionality), not distance, when comparing movie ratings. Second, cosine similarity can handle sparse data better than euclidean. Unfortunately, cosine similarity does not take magnitude into consideration, so a "Love" and "Like" rating for a movie would be considered the same. A metric that takes directionality and magnitude into consideration might be better. The data will be a sparse matrix, so this also makes cosine similarity superior to euclidean. We chose the k neighbors with the highest similarity rating.



**new version 2**
For this problem, we care about directionality (liked the movie or not) and magnitude (love or liked movie). If we just cared about directionality, cosine similarity would be a good choice, as it would measure the angles between the two users' movie rating vectors. But since we care about the magnitude as well, Pearson coefficient should be used. This similarity metric captures both directionality and magnitude.



**Step 3: K-nearest neighbors**
Using ______________ as the distance metric, the similarities of a user with each other user is calculated. Then we sort the most similar users in descending order. The top k users will then be selected. These are the users that have the most similar preferences to the target user. The determination of the value of k is the most important step. A higher K will capture more diverse preferences but may also introduce noise, while a lower K may lead to more personalized recommendations but could be sensitive to outliers.


**Step 4: Recommendation System (Collaborative Filter)**
The weighted sum is calculated by summing the product of the value of cosine similarity and the rating posted by the neighbor. The total weight is just the sum of absolute similarity per neighbor. With the above two values, we estimate the rating by dividing the weighted sum by the total weight. This is how a movie is recommended to the target user.

=============>(RK ???)



**Pseudo-code:**
function collaborative_filter(user_id, movie_id, k_neighbors):
    similar_users = find_k_nearest_neighbors(user_id, k_neighbors)  # Find k nearest neighbors
    weighted_sum = 0
    total_weight = 0
    
    for neighbor_id in similar_users:
        if neighbor_has_rated_movie(neighbor_id, movie_id):
      	similarity = compute_similarity(user_id, neighbor_id)  # Compute similarity between users
            rating = get_rating(neighbor_id, movie_id)  # Get neighbor's rating for the movie
            weighted_sum += similarity * rating
            total_weight += abs(similarity)
    
    if total_weight != 0:
        estimated_rating = weighted_sum / total_weight
    else:
        estimated_rating = default_rating  # Default value if no similar users have rated the movie
    
    return estimated_rating
