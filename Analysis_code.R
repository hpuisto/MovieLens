#######################################################################
# HarvardX PH125.9x Data Science: Capstone
# MovieLens Project
# Author: Hannah Puisto
# Date: December 2023
#
# Github: https://github.com/hpuisto/movielens
##########################################################

##########################################################
# Import data and create datasets
# This first section of code was provided by the course
##########################################################

##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##########################################################
##########################################################
# Begin personal code
##########################################################
##########################################################

##########################################################
# Initial data exploration
##########################################################

# Number of columns and rows in edx and final_holdout_test datasets
dim(edx)
dim(final_holdout_test)

# Number of zero ratings given for each dataset
edx %>% filter(rating == 0) %>% nrow()
final_holdout_test %>% filter(rating == 0) %>% nrow()

# Number of different movies in each dataset
edx %>% group_by(movieId) %>% summarize(count = n())
final_holdout_test %>% group_by(movieId) %>% summarize(count = n())

# Number of different users in each dataset
edx %>% group_by(userId) %>% summarize(count = n())
final_holdout_test %>% group_by(userId) %>% summarize(count = n())

# Number of movie ratings by genre for each dataset
# View of all unique genres
unique_genres_list_edx <- str_extract_all(unique(edx$genres), "[^|]+") %>%
  unlist() %>%
  unique()
unique_genres_list_test <- str_extract_all(unique(final_holdout_test$genres), "[^|]+") %>%
  unlist() %>%
  unique()
# Summarize
edx %>% summarize(n())
final_holdout_test %>% summarize(n())

# Movies with the greatest number of ratings for each dataset
edx %>% group_by(title) %>%
  summarize(n_ratings = n()) %>%
  arrange(desc(n_ratings))
final_holdout_test %>% group_by(title) %>%
  summarize(n_ratings = n()) %>%
  arrange(desc(n_ratings))

# Most common ratings given for each dataset
edx %>% group_by(rating) %>%
  summarize(n_ratings = n()) %>%
  arrange(desc(n_ratings))
final_holdout_test %>% group_by(rating) %>%
  summarize(n_ratings = n()) %>%
  arrange(desc(n_ratings))

##########################################################
# Use overall average rating
##########################################################

# Calculate overall average rating for the training dataset
mu <- mean(edx$rating)

# Predict unknown ratings with mu and calculate the RMSE
RMSE(final_holdout_test$rating, mu)
