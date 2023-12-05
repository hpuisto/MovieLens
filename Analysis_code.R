##########################################################
# HarvardX PH125.9x Data Science: Capstone
# Project 1: MovieLens
# Author: Hannah Puisto
# Date: December 2023
#
# Github: https://github.com/hpuisto/movielens
##########################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(terra)) install.packages("terra", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(knitr)
library(terra)
library(data.table)

##########################################################
# Initial data exploration
##########################################################

# The RMSE function that will be used in this project is:
RMSE <- function(true_ratings = NULL, predicted_ratings = NULL) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

##########################################################
# Number of columns and rows in edx and final_holdout_test datasets
dim(edx)
dim(final_holdout_test)

##########################################################
# Number of zero ratings given for each dataset
edx %>% filter(rating == 0) %>% nrow()
final_holdout_test %>% filter(rating == 0) %>% nrow()

##########################################################
# Number of different movies in edx dataset
edx %>% group_by(movieId) %>% summarize(count = n())

# Histogram of ratings per movie for edx dataset
hist_movies <- edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 40, fill = "black") +
  labs(
    title = "Histogram of Ratings per Movie",
    x = "Movie ID", y = "Number of Ratings", fill = element_blank()
  ) +
  theme_classic()
hist_movies

##########################################################
# Number of different users in edx dataset
edx %>% group_by(userId) %>% summarize(count = n())

# Histogram of ratings per user for edx dataset
hist_users <- edx %>%
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 40, fill = "black") + 
  labs(title = "Histogram of Ratings per User",
       x = "User ID", y = "Number of Ratings", fill = element_blank()) +
  theme_classic()
hist_users

##########################################################
# Number of each type of rating in edx dataset
edx %>% group_by(rating) %>% summarize(count = n())

# Histogram of Ratings
hist_rate <- edx %>%
  ggplot(aes(rating)) +
  geom_histogram(fill = "black") +
  labs(
    title = "Histogram of Ratings",
    x = "Rating", y = "Count", fill = element_blank()
  ) +
  theme_classic()
hist_rate

##########################################################
# Number of movie ratings by genre for edx dataset
# List of all unique genres
unique_genres_list <- str_extract_all(unique(edx$genres), "[^|]+") %>%
  unlist() %>%
  unique()
unique_genres_list

# Create an extended version of both the edx and final_holdout_test datasets. This separates the genres for a movie listed and produces a new row for each unique genre listed for that movie (multiple rows for the same user and movie will exist if more than one genre was listed for a given movie)
edx_genres <- edx %>%
  separate_rows(genres, sep = "\\|", convert = TRUE)

test_genres <- final_holdout_test %>%
  separate_rows(genres, sep = "\\|", convert = TRUE)

# Histogram of ratings per genre for edx dataset
hist_genres <- ggplot(edx_genres, aes(x = reorder(genres, genres, function(x) - length(x)))) +
  geom_bar(fill = "black") +
  labs(title = "Histogram of Ratings per Genre",
       x = "Genre", y = "Number of Ratings") +
  scale_y_continuous(labels = paste0(1:4, "M"),
                     breaks = 10^6 * 1:4) +
  coord_flip() +
  theme_classic()
hist_genres

##########################################################
# Movies with the greatest number of ratings for edx dataset
edx %>% group_by(title) %>%
  summarize(n_ratings = n()) %>%
  arrange(desc(n_ratings))

##########################################################
# Most common ratings given for edx dataset
edx %>% group_by(rating) %>%
  summarize(n_ratings = n()) %>%
  arrange(desc(n_ratings))

##########################################################
##########################################################
# Begin Analysis
##########################################################
##########################################################

##########################################################
# Method 1: Use overall average rating to predict unknown rating
##########################################################

# Calculate overall average rating for the edx dataset
mu <- mean(edx$rating)

# Predict unknown ratings with mu and calculate the RMSE
RMSE(final_holdout_test$rating, mu)

######################
# Method 2: Use a movie bias (b) to average the rankings for each movie (m)
######################

# Create an movie bias term, b_m
b_m <- edx %>%
  group_by(movieId) %>%
  summarize(b_m = mean(rating - mu))

# Predict unknown ratings for each movie on test dataset with mu and b_m
predicted_ratings <- final_holdout_test %>% 
  left_join(b_m, by='movieId') %>%
  mutate(pred = mu + b_m) %>%
  pull(pred)

# Calculate RMSE of movie bias effect
RMSE(final_holdout_test$rating, predicted_ratings)

###############################
# Method 3: Use movie bias (b_m) and add a user bias (b) to average the rankings for each user (u)
###############################

# Create user bias term, b_u
b_u <- edx %>% 
  left_join(b_m, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_m))

# Predict new ratings for each movie on test dataset using both movie and user biases
predicted_ratings <- final_holdout_test %>% 
  left_join(b_m, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  mutate(pred = mu + b_m + b_u) %>%
  pull(pred)

# Calculate RMSE of movie and user biases effect
RMSE(final_holdout_test$rating, predicted_ratings)


###########################################
# Method 4: Use movie bias (b_m), user bias (b_u), and add genre bias (b) to average the rankings for each genre (g)
###########################################

# Create genre bias term, b_g, on extended version of edx: edx_genres calculated above
b_g <- edx_genres %>% 
  left_join(b_m, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_m - b_u))

# Predict new ratings for each movie on extended test dataset (test_genres) using both movie and user biases
predicted_ratings <- test_genres %>% 
  left_join(b_m, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(b_g, by='genres') %>%
  mutate(pred = mu + b_m + b_u + b_g) %>%
  pull(pred)

# Calculate RMSE of movie, user, and genre biases effect
RMSE(test_genres$rating, predicted_ratings)

###########################################
# Method 5: Use regularization on movie and user biases effect to reduce errors
###########################################

# Determine best lambda from a sequence
lambdas <- seq(from=0, to=10, by=0.1)

# Output RMSE of each lambda by repeating earlier steps (with regularization)
rmses <- sapply(lambdas, function(l){
  # Calculate overall average rating for the edx dataset
  mu <- mean(edx$rating)
  # Compute regularized movie bias term
  b_m <- edx %>% 
    group_by(movieId) %>%
    summarize(b_m = sum(rating - mu)/(n()+l))
  # Compute regularized user bias term
  b_u <- edx %>% 
    left_join(b_m, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_m - mu)/(n()+l))
  # Compute regularized genre bias term on extended version of edx
  b_g <- edx_genres %>% 
    left_join(b_m, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - b_m - b_u)/(n()+l))
  predicted_ratings <- test_genres %>% 
    left_join(b_m, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_g, by='genres') %>%
    mutate(pred = mu + b_m + b_u + b_g) %>%
    pull(pred)
  # Output RMSE of these predictions
  return(RMSE(test_genres$rating, predicted_ratings))
})

# Plot of RMSE vs lambdas
plot_rmses <- ggplot(data.frame(lambdas, rmses), aes(lambdas, rmses)) + geom_line()
# Print minimum RMSE 
min(rmses)


##########################################################
##########################################################
# Results
##########################################################
##########################################################

######################################################
# Final model: Regularized movie, user, and genre bias effects
######################################################

# The final linear model with the minimizing lambda
lam <- lambdas[which.min(rmses)]

# Compute final regularized movie bias term
b_m <- edx %>% 
  group_by(movieId) %>%
  summarize(b_m = sum(rating - mu)/(n()+lam))

# Compute final regularized user bias term
b_u <- edx %>% 
  left_join(b_m, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_m - mu)/(n()+lam))

# Compute final regularized genre bias term
b_g <- edx_genres %>% 
  left_join(b_m, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - b_m - b_u)/(n()+lam))

# Compute final predictions on final_holdout_test set based on the above biases
predicted_ratings <- test_genres %>% 
  left_join(b_m, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(b_g, by='genres') %>%
  mutate(pred = mu + b_m + b_u + b_g) %>%
  pull(pred)

# Output final RMSE of these predictions
RMSE(test_genres$rating, predicted_ratings)
