# EdX Capstone Project 1
## MovieLens
This is the first project in the HarvardX Data Science Capstone course. The goal of this project was to predict movie ratings using a large dataset defined by the course. The submission for the MovieLens project requires three files: a report in the form of an Rmd file, a report in the form of a PDF document knit from your Rmd file, and an R script that generates our predicted movie ratings and calculates RMSE. The grade for the project will be based on two factors: 1. The report and script (75%), and 2. The RMSE returned by testing our algorithm on the validation set (25%).

## Important files in this repository
1. `getData.R` was provided as part of the project setup in the course prompt. This file retrieves the data we need to perform our analysis and splits it into a "training" set named `edx` and a validation set named `final_holdout_test` to be used at the very end of our analysis to measure the accuracy of our work.
2. `Analysis_code.R` leverages `getData.R` to import and save the data, and then proceeds with our exploration and analysis. The last portion of this script applies the final model to the `final_holdout_test` set for a final measure of accuracy.
3. `report.Rmd` is an R-markdown file that generates a final report.
4. `Report.pdf` is the final report generated for this project.
