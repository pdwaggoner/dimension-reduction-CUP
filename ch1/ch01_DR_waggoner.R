# Nonparametric Dimension Reduction for High Dimensional Data (Cambridge University Press)
# Philip D. Waggoner, pdwaggoner@uchicago.edu

# Section 1: Introduction and data preprocessing


# Packages needed for this section:
install.packages(c("tidyverse", "here"))

# Load libraries
library(tidyverse)
library(here)


# Read in ANES 2019 Pilot data - emphasize benefit of using R projects to keep things organized, e.g., allowing for use of "here" package, like so...
anes_start <- read_csv(here("Data", "anes_pilot_2019.csv"))

## FTs used (35 in total)
#     fttrump     ftobama     ftbiden     ftwarren    ftsanders   ftbuttigieg
#     ftharris    ftblack     ftwhite     fthisp      ftasian     ftmuslim   
#     ftillegal   ftimmig1    ftimmig2    ftjournal   ftnato      ftun       
#     ftice       ftnra       ftchina     ftnkorea    ftmexico    ftsaudi    
#     ftukraine   ftiran      ftbritain   ftgermany   ftjapan     ftisrael   
#     ftfrance    ftcanada    ftturkey    ftrussia    ftpales   

# Feature Engineering and Data Management


# New packages/libraries
library(skimr)
library(naniar)
library(recipes)
library(tictoc)
library(knitr)

# Cleaning (4 main steps)
anes_raw <- anes_start %>%
  select(pid7, fttrump, ftobama, ftbiden, ftwarren, ftsanders, ftbuttigieg, 
         ftharris, ftblack, ftwhite, fthisp, ftasian, ftmuslim,   
         ftillegal, ftimmig1, ftimmig2, ftjournal, ftnato, ftun,       
         ftice, ftnra, ftchina, ftnkorea, ftmexico, ftsaudi,    
         ftukraine, ftiran, ftbritain, ftgermany, ftjapan, ftisrael,   
         ftfrance, ftcanada, ftturkey, ftrussia, ftpales) %>% 
  mutate(democrat = case_when(pid7 == 1 ~ 1, 
                              pid7 == 2 ~ 1, 
                              pid7 == 3 ~ 1, 
                              pid7 == 4 ~ 0, 
                              pid7 == 5 ~ 0, 
                              pid7 == 6 ~ 0, 
                              pid7 == 7 ~ 0, 
                              pid7 == 8 ~ 0), 
         Trump = replace(fttrump, fttrump > 100 | fttrump < 0, NA), 
         Obama = replace(ftobama, ftobama > 100 | ftobama < 0, NA),
         Biden = replace(ftbiden, ftbiden > 100 | ftbiden < 0, NA),
         Warren = replace(ftwarren, ftwarren > 100 | ftwarren < 0, NA),
         Sanders = replace(ftsanders, ftsanders > 100 | ftsanders < 0, NA),
         Buttigieg = replace(ftbuttigieg, ftbuttigieg > 100 | ftbuttigieg < 0, NA),
         Harris = replace(ftharris, ftharris > 100 | ftharris < 0, NA),
         Black = replace(ftblack, ftblack > 100 | ftblack < 0, NA),
         White = replace(ftwhite, ftwhite > 100 | ftwhite < 0, NA),
         Hispanic = replace(fthisp, fthisp > 100 | fthisp < 0, NA),
         Asian = replace(ftasian, ftasian > 100 | ftasian < 0, NA),
         Muslim = replace(ftmuslim, ftmuslim > 100 | ftmuslim < 0, NA),
         Illegal = replace(ftillegal, ftillegal > 100 | ftillegal < 0, NA),
         Immigrants = replace(ftimmig1, ftimmig1 > 100 | ftimmig1 < 0, NA),
         `Legal Immigrants` = replace(ftimmig2, ftimmig2 > 100 | ftimmig2 < 0, NA),
         Journalists = replace(ftjournal, ftjournal > 100 | ftjournal < 0, NA),
         NATO = replace(ftnato, ftnato > 100 | ftnato < 0, NA),
         UN = replace(ftun, ftun > 100 | ftun < 0, NA),
         ICE = replace(ftice, ftice > 100 | ftice < 0, NA),
         NRA = replace(ftnra, ftnra > 100 | ftnra < 0, NA),
         China = replace(ftchina, ftchina > 100 | ftchina < 0, NA),
         `North Korea` = replace(ftnkorea, ftnkorea > 100 | ftnkorea < 0, NA),
         Mexico = replace(ftmexico, ftmexico > 100 | ftmexico < 0, NA),
         `Saudi Arabia` = replace(ftsaudi, ftsaudi > 100 | ftsaudi < 0, NA),
         Ukraine = replace(ftukraine, ftukraine > 100 | ftukraine < 0, NA),
         Iran = replace(ftiran, ftiran > 100 | ftiran < 0, NA),
         Britain = replace(ftbritain, ftbritain > 100 | ftbritain < 0, NA),
         Germany = replace(ftgermany, ftgermany > 100 | ftgermany < 0, NA),
         Japan = replace(ftjapan, ftjapan > 100 | ftjapan < 0, NA),
         Israel = replace(ftisrael, ftisrael > 100 | ftisrael < 0, NA),
         France = replace(ftfrance, ftfrance > 100 | ftfrance < 0, NA),
         Canada = replace(ftcanada, ftcanada > 100 | ftcanada < 0, NA),
         Turkey = replace(ftturkey, ftturkey > 100 | ftturkey < 0, NA),
         Russia = replace(ftrussia, ftrussia > 100 | ftrussia < 0, NA),
         Palestine = replace(ftpales, ftpales > 100 | ftpales < 0, NA)
  ) %>% 
  glimpse() 

# drop the original pid7 for ease
anes_raw <- anes_raw %>% 
  select(-c(pid7, fttrump, ftobama, ftbiden, ftwarren, ftsanders, ftbuttigieg, 
            ftharris, ftblack, ftwhite, fthisp, ftasian, ftmuslim,   
            ftillegal, ftimmig1, ftimmig2, ftjournal, ftnato, ftun,       
            ftice, ftnra, ftchina, ftnkorea, ftmexico, ftsaudi,    
            ftukraine, ftiran, ftbritain, ftgermany, ftjapan, ftisrael,   
            ftfrance, ftcanada, ftturkey, ftrussia, ftpales))

# Feature Engineering and Missing Data
# Start by visualizing patterns of missingness. There are many ways to do so, e.g...

# check where missing data exist in the full data space (feature-level)
anes_raw %>% 
  select(order(desc(colnames(.)))) %>% 
  gg_miss_which() + 
  labs(caption = "Note: Gray = Missing, Black = Complete")

# cumulative missingness over observations
anes_raw %>% 
  gg_miss_case_cumsum()

# missing data by feature
## cumulative
anes_raw %>% 
  gg_miss_var()

## or percentage
anes_raw %>% 
  gg_miss_var(show_pct = TRUE)

## can also facet within call
anes_raw %>% 
  gg_miss_var(democrat) 

# intersection
anes_raw %>% 
  gg_miss_upset()

# can drop...
anes_deleted <- anes_raw %>% 
  drop_na()

# or impute...
# first, build the recipe
recipe <- recipe(democrat ~ ., 
                 data = anes_raw) %>%
  step_knnimpute(all_predictors())

# now, impute (note: wrap the few functions in brackets to run the full chunk at once)
{ 
  tic()
  anes_imputed <- prep(recipe) %>% 
    juice()
  toc()
  } # ~ 3 seconds on 4 cores

# double check completion rates for before and after imputation
skim(anes_raw) # before
skim(anes_imputed) # after

anes_imputed %>% 
  skim() %>% 
  kable(format = 'latex') 

## a closer look at top three missing features: ftimmig1, ftimmig2, and ftbuttigieg

skim(anes_raw$Immigrants) # before
skim(anes_imputed$Immigrants) # after

skim(anes_raw$`Legal Immigrants`) # before
skim(anes_imputed$`Legal Immigrants`) # after

skim(anes_raw$Buttigieg) # before
skim(anes_imputed$Buttigieg) # after

# proceed with the clean data obj
anes <- anes_imputed

# store for later use
saveRDS(anes, 
        file = "anes.rds")
