# Nonparametric Dimension Reduction for High Dimensional Data (Cambridge University Press)
# Philip D. Waggoner, pdwaggoner@uchicago.edu

# Section 5: Neural Network-Based Approaches


# Packages needed for this section:
install.packages(c(...))

# Load libraries
library(tidyverse)
library(here)
library(amerika)
library(tictoc)
library(kohonen)


# First, read in cleaned and preprocessed 2019 ANES Pilot Data
anes <- read_rds(here("Data", "anes.rds"))

# let's jump in

set.seed(1234)

# SOM first, then AE
# for SOM: We set the predeterminied number of neurons/nodes and search over the space.
# standardize
anes_scaled <- anes[ ,1:35] %>% 
  scale()

# fit the SOM
# specify the dimensions of the grid to search
search_grid <- somgrid(xdim = 10, 
                       ydim = 10, 
                       topo = "rectangular",
                       neighbourhood.fct = "gaussian") 

# train the SOM based on the grid specifications
{
  tic()
som_fit <- som(anes_scaled,
               grid = search_grid,
               alpha = c(0.1, 0.001), # learning rate; (default vals decline from .05 to .01)
               radius = 1,# neighborhood size
               rlen = 500, # iterations
               dist.fcts = "euclidean", 
               mode = "batch") 
  toc()
} # ~12.275 seconds

# plot the training progress
som_fit$changes %>% 
  as_tibble() %>% 
  rename(., changes = V1) %>% 
  mutate(., iteration = seq(1:length(changes))) %>% 
  ggplot(aes(iteration, changes)) +
  geom_line() +
  labs(x = "Training Iteration",
       y = "Mean Distance to Closest Node") +
  theme_minimal()+
  theme(axis.title = element_text(size=15),
        axis.text = element_text(size=17),
        legend.text = element_text(size=13),
        legend.title = element_text(size=15))

# Classification assignments from SOM (and some compared to other clustering like k-means (hard part) and FCM (soft part))
point_colors <- c(amerika_palettes$Republican[2], 
                  amerika_palettes$Democrat[2])

neuron_colors <- c(amerika_palettes$Republican[3], 
                   amerika_palettes$Democrat[3])

## find and plot cluster assignments via k-means
kmeans_clusters <- som_fit$codes[[1]] %>% 
  kmeans(., centers = 2)

class_assign_km <- map_dbl(kmeans_clusters$cluster, ~{
  if(. == 1) 2
  else 1
}
)

# viz k-means cluster assignments
plot(som_fit, 
     type = "mapping", 
     pch = 21, 
     bg = point_colors[as.factor(anes$democrat)],
     shape = "straight",
     bgcol = neuron_colors[as.integer(class_assign_km)],
     main = " "); add.cluster.boundaries(x = som_fit, clustering = class_assign_km, 
                                                      lwd = 5, lty = 5)

## find and plot cluster assignments via fcm
fcm_clusters <- som_fit$codes[[1]] %>% 
  ppclust::fcm(., centers = 2)

class_assign_fcm <- map_dbl(fcm_clusters$cluster, ~{
  if(. == 1) 2
  else 1
}
)

# viz fcm cluster assignments
plot(som_fit, 
     type = "mapping", 
     pch = 21, 
     bg = point_colors[as.factor(anes$democrat)],
     shape = "straight",
     bgcol = neuron_colors[as.integer(class_assign_fcm)],
     main = " "); add.cluster.boundaries(x = som_fit, clustering = class_assign_fcm, 
                                                      lwd = 5, lty = 5)

# numeric comparison of centroid coordinates
#(fcm <- fcm_clusters$v[order(fcm_clusters$v[ , 1]), ])
#(kmeans <- kmeans_clusters$centers[order(kmeans_clusters$centers[, 1]), ])
#round(kmeans - fcm, 2) 

## find and plot cluster assignments via HAC (KEEP?)
#hac_clusters <- som_fit$codes[[1]] %>% 
#  dist() %>% 
#  hclust() %>% 
#  cutree(., k = 2)

#class_assign <- map_dbl(hac_clusters, ~{
#  if(. == 1) 2
#  else 1
#}
#)

# viz HAC cluster assignments
#plot(som_fit, type = "mapping", pch = 21, 
#     bg = point_colors[as.factor(anes$democrat)],
#     shape = "straight",
#     bgcol = neuron_colors[as.integer(class_assign)],
#     main = "2 clusters via\nHierarchical Agglomerative Clustering"); add.cluster.boundaries(x = som_fit, 
#                                                  clustering = class_assign, 
#                                                  lwd = 5, 
#                                                  lty = 5)


#


# We can plot SOM results in many other ways like, for example, summed distances between neighbors in the grid space (type = "dist.neighbours") or the mean distances between units and observations (type = "quality"). Though these diagnostic plots are useful, for the sake of space I point readers to the documentation (e.g., ?plot.kohonen), and focus instead on tying results back to the substantive social science example of understanding latent partisan structure in feeling thermometers. To this end, I will focus on the codes produced by the tuned SOM. Codes, which are sometimes called weight vectors (called via $codes), represent the representation of each feature in each node. Substantively, codes give a sense of the specific features that are attracted to similar nodes, such that we are able to pick up on grouping across features in the SOM grid space. For example, like features (e.g., feelings toward Barack Obama and Elizabeth Warren) might heavily characterize ("weight") a node, compared to other features (e.g., feelings toward Donald Trump). The result is a picture of the relationships between both features and nodes across the full input space. 

# A simple plot of codes via the kohonen package might look something like,

plot(som_fit, 
     type = "codes")

# However, in our case, we have a relatively high dimensional input space with 35 dimensions (feeling thermometers) at play. Thus, running the previous line will generate the full grid with uninterpretable ranges of weighted values in each node. With out feature labels, conditional colors, and so on given the massive amount of information crammed into a single space, this type of plot is relatively useless (note: for spaces with fewer inputs, say, e.g., 4, the codes plot defaults to a "fan diagram", with larger slices indicating greater weight in a given node by the given feature. I encourage readers to constrain the space and make it smaller to see the default option for codes plots). As such, we will proceed on a feature-by-feature basis, and tie the results back into a substantive understanding and exploration of the American political preference space from the ANES. Such a use of codes from our fit SOM results in a slightly nuanced view of these weight vectors compared to more common interpretation discussed above. Substantively, in this case, we can think of weight vectors/codes like we might correlations between features and neurons. Features that are more similar will trend toward neurons in a positive direction, suggesting similar grouping and thus latent structure across those features. This is compared to features that are more different from each other, which will trend negatively. 

# For example, consider the direction of the codes between feelings toward Trump and Obama (we might expect a negative relationship/different information across these and that's what we see). Inversely, we can see that features that we might expect to be more similar to each other or picking up common structure to be trending in a positive direction. For example, we can see this positive trend in feelings toward Bernie Sanders and Barack Obama. 

# trump and obama (negative relationship)
som_fit$codes %>% 
  as.data.frame() %>% 
  ggplot(aes(Trump, Obama)) +
  geom_point() +
  geom_smooth(method = "loess", se = FALSE) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_vline(xintercept = 0, linetype = "dashed") +
 # labs(title = "SOM Codes for Feelings toward Trump and Obama") +
  theme_minimal()+
  theme(axis.title = element_text(size=15),
        axis.text = element_text(size=17),
        legend.text = element_text(size=13),
        legend.title = element_text(size=15))

# sanders and obama (positive relationship)
som_fit$codes %>% 
  as.data.frame() %>% 
  ggplot(aes(Sanders, Obama)) +
  geom_point() +
  geom_smooth(method = "loess", se = FALSE) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_vline(xintercept = 0, linetype = "dashed") +
 # labs(title = "SOM Codes for Feelings toward Sanders and Obama") +
  theme_minimal()+
  theme(axis.title = element_text(size=15),
        axis.text = element_text(size=17),
        legend.text = element_text(size=13),
        legend.title = element_text(size=15))

# Though these sorts of base, naive expectations are easier for people, based on what we know about partisan politics and ideological preferences in American politics, the same is true for naive expectations across institutions, like we might expect negative across UN and the NRA, but positive across ICE and the NRA. These are the patterns we see, suggesting the SOM is picking up this latent, partisan structure in these data in line with substantive/domain expectations.

# UN and NRA (negative relationship)
som_fit$codes %>% 
  as.data.frame() %>% 
  ggplot(aes(UN, NRA)) +
  geom_point() +
  geom_smooth(method = "loess", se = FALSE) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_vline(xintercept = 0, linetype = "dashed") +
  #labs(title = "SOM Codes for Feelings toward the UN and NRA") +
  theme_minimal()+
  theme(axis.title = element_text(size=15),
        axis.text = element_text(size=17),
        legend.text = element_text(size=13),
        legend.title = element_text(size=15))

# UN and NRA (positive relationship)
som_fit$codes %>% 
  as.data.frame() %>% 
  ggplot(aes(ICE, NRA)) +
  geom_point() +
  geom_smooth(method = "loess", se = FALSE) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_vline(xintercept = 0, linetype = "dashed") +
  #labs(title = "SOM Codes for Feelings toward ICE and the NRA") +
  theme_minimal()+
  theme(axis.title = element_text(size=15),
        axis.text = element_text(size=17),
        legend.text = element_text(size=13),
        legend.title = element_text(size=15))

#



## AE HERE...

## BEFORE APPLICATION, consider talking about the use of ML engines like h20 (need JDK for this to work), tensorflow, and keras... we will be using h20 for this application section given it's power; plus its a good skill to develop (working with cloud computing and external engines, even locally as we are doing here)

## ALSO, here you need to mention the common ML approach to modeling of training, testing, validation, and tuning

library(h2o)
library(doParallel)

set.seed(1234)

# First, convert party to factor for modeling 
anes$democrat <- factor(anes$democrat)

# initializing the H2O cluster/session
my_h2o <- h2o.init()

# Detecting the available number of cores; set up cluster for parallel session
cores <- detectCores() - 1 # leave one for the rest of the computer to process normally
cluster <- makeCluster(cores, setup_timeout = 0.5)
registerDoParallel(cluster) # takes about 5 minutes to set up the first time

# Create an H2O dataframe
anes_h2o <- anes %>% 
  as.h2o()

# Create train (0.60), validation (0.20), and test (0.20) sets 
split_frame <- h2o.splitFrame(anes_h2o, 
                              ratios = c(0.6, 0.2), 
                              seed = 1234)   

# a quick look
split_frame %>% 
  glimpse()

train <- split_frame[[1]]
validation <- split_frame[[2]]
test <- split_frame[[3]]

# Store response and predictors separately (per h2o syntax)
response <- "democrat"

predictors <- setdiff(colnames(train), response)

# Construct vanilla autoencoder with tanh activation and a single hidden layer with 16 nodes
{
  tic()
autoencoder <- h2o.deeplearning(x = predictors, 
                                training_frame = train,
                                autoencoder = TRUE,
                                reproducible = TRUE,
                                seed = 1234,
                                hidden = c(16), # less than half of the feature space (35 total; so 16)
                                epochs = 100,
                                activation = "Tanh",
                                validation_frame = test)
  toc()
} # ~5.3 seconds

# Save the model, if desired
#h2o.saveModel(autoencoder, 
#              path = "autoencoder", 
#              force = TRUE)

# load the model directly, if desired
#autoencoder <- h2o.loadModel(".../file/path/here")

# we can make predictions of reconstruction on test set (but we will come back to this in a bit with a more targeted task via predicting PID)
#preds <- h2o.predict(autoencoder, test)

# Let's extract the codings/features from fit AE - note the similar terminology with SOM
codings_train <- h2o.deepfeatures(autoencoder, 
                                  data = train, 
                                  layer = 1) %>% # "layer" is referring to the number of hidden layers (1 in our case)
  as.data.frame() %>%
  mutate(democrat = as.vector(train[ , 36]))
      ## these are read as, e.g., DF.L1.C1 - "data frame, layer number, column number"

# Numeric inspection of the "codes" (or, "scores")
codings_train %>% 
  head(10)


# Visual inspection
# Substantively: checking to see whether our AE has detected the party labels or not over the first two deep features
{
p1 <- ggplot(codings_train, aes(x = DF.L1.C1, 
                                y = DF.L1.C2, 
                                color = factor(democrat))) +
  geom_point(alpha = 0.6) + 
  stat_ellipse() +
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Party",
                     breaks=c("0", "1"),
                     labels=c("Non-Democrat", "Democrat")) +
  labs(title = "Deep Features 1 & 2",
       color = "Democrat") + 
  theme_minimal()

# (3 and 4)
p2 <- ggplot(codings_train, aes(x = DF.L1.C3, 
                                y = DF.L1.C4, 
                                color = factor(democrat))) +
  geom_point(alpha = 0.6) + 
  stat_ellipse() +
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Party",
                     breaks=c("0", "1"),
                     labels=c("Non-Democrat", "Democrat")) +
  labs(title = "Deep Features 3 & 4",
       color = "Democrat") + 
  theme_minimal()

# 5 & 6
p3 <- ggplot(codings_train, aes(x = DF.L1.C5, 
                                y = DF.L1.C6, 
                                color = factor(democrat))) +
  geom_point(alpha = 0.6) + 
  stat_ellipse() +
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Party",
                     breaks=c("0", "1"),
                     labels=c("Non-Democrat", "Democrat")) +
  labs(title = "Deep Features 5 & 6",
       color = "Democrat") + 
  theme_minimal()

# 7 & 8
p4 <- ggplot(codings_train, aes(x = DF.L1.C7, 
                                y = DF.L1.C8, 
                                color = factor(democrat))) +
  geom_point(alpha = 0.6) + 
  stat_ellipse() +
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Party",
                     breaks=c("0", "1"),
                     labels=c("Non-Democrat", "Democrat")) +
  labs(title = "Deep Features 7 & 8",
       color = "Democrat") + 
  theme_minimal()

# 9 & 10
p5 <- ggplot(codings_train, aes(x = DF.L1.C9, 
                                y = DF.L1.C10, 
                                color = factor(democrat))) +
  geom_point(alpha = 0.6) + 
  stat_ellipse() +
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Party",
                     breaks=c("0", "1"),
                     labels=c("Non-Democrat", "Democrat")) +
  labs(title = "Deep Features 9 & 10",
       color = "Democrat") + 
  theme_minimal()

# 11 & 12
p6 <- ggplot(codings_train, aes(x = DF.L1.C11, 
                                y = DF.L1.C12, 
                                color = factor(democrat))) +
  geom_point(alpha = 0.6) + 
  stat_ellipse() +
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Party",
                     breaks=c("0", "1"),
                     labels=c("Non-Democrat", "Democrat")) +
  labs(title = "Deep Features 11 & 12",
       color = "Democrat") + 
  theme_minimal()

# 13 & 14
p7 <- ggplot(codings_train, aes(x = DF.L1.C13, 
                                y = DF.L1.C14, 
                                color = factor(democrat))) +
  geom_point(alpha = 0.6) + 
  stat_ellipse() +
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Party",
                     breaks=c("0", "1"),
                     labels=c("Non-Democrat", "Democrat")) +
  labs(title = "Deep Features 13 & 14",
       color = "Democrat") + 
  theme_minimal()

# 15 & 16
p8 <- ggplot(codings_train, aes(x = DF.L1.C15, 
                                y = DF.L1.C16, 
                                color = factor(democrat))) +
  geom_point(alpha = 0.6) + 
  stat_ellipse() +
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Party",
                     breaks=c("0", "1"),
                     labels=c("Non-Democrat", "Democrat")) +
  labs(title = "Deep Features 15 & 16",
       color = "Democrat") + 
  theme_minimal()

# view together
library(patchwork)

(p1 + p2 + p3 + p4) / 
  (p5 + p6 + p7 + p8)
}

# Great, we have built an AE and plotted the deep features, with color by PID and definitely picked up similar structure as with the previous methods (clearly groups at either extreme, and blending near the middle, reflecting American political dynamics, at a high, intuitive level). But let's go deeper! 


#   1. anomaly detection via reconstruction error - CLEAN UP?? KEEP???
# Compute reconstruction error for anomaly detection (MSE between output and input layers)
#recon_error <- h2o.anomaly(autoencoder, test) %>% 
#  as.data.frame()

#plot.ts(recon_error)#

#test_recon <- h2o.predict(autoencoder, test)
#head(test_recon)


#  Make some predictions (a supervised task) + feature importance
## First, let's build a new model to predict party affiliation as a function of the deep features
# With our deep features from the AE, train a deep neural net (2 HL, total of 10 nodes per the rule)
codings_val <- h2o.deepfeatures(object = autoencoder, 
                                data = validation, 
                                layer = 1) %>%
  as.data.frame() %>%
  mutate(democrat = as.factor(as.vector(validation[ , 36]))) %>%
  as.h2o()

# Store using new codings_val object
deep_features <- setdiff(colnames(codings_val), response)

deep_net <- h2o.deeplearning(y = response,
                             x = deep_features,
                             training_frame = codings_val,
                             reproducible = TRUE, 
                             ignore_const_cols = FALSE,
                             seed = 1234,
                             hidden = c(8, 8), 
                             epochs = 100,
                             activation = "Tanh")

## Make predictions & classify
test_3 <- h2o.deepfeatures(object = autoencoder, 
                           data = test, 
                           layer = 1)

test_pred <- h2o.predict(deep_net, test_3, type = "response") %>%
  as.data.frame() %>%
  mutate(truth = as.vector(test[, 36]))

# Visualize predictions
#test_pred %>% 
#  head(25)

# Summarize predictions as confusion matrix
print(h2o.predict(deep_net, test_3) %>%
        as.data.frame() %>%
        mutate(truth = as.vector(test[, 36])) %>%
        group_by(truth, predict) %>%
        summarise(n = n()) %>%
        mutate(freq = n / sum(n)))

table(h2o.predict(deep_net, test_3))

# Not too bad! (remember to go back through and clear up exactly why each of these is constructed as it is and why and how they fit together.)


## Feature importance


# calc first
fimp <- as.data.frame(h2o.varimp(deep_net)) %>% 
  arrange(desc(relative_importance))

# viz relative
fimp %>% 
  ggplot(aes(x = relative_importance, 
             y = reorder(variable, -relative_importance))) +
  geom_point(color = "dark red", 
             fill = "dark red", 
             alpha = 0.5,
             size = 5) +
  labs(#title = "Relative Feature Importance",
       #subtitle = "Deep Neural Network (2 hidden layers with 16 total neurons)",
       x = "Relative Importance",
       y = "Feature") + 
  theme_minimal()+
  theme(axis.title = element_text(size=15),
        axis.text = element_text(size=17),
        legend.text = element_text(size=13),
        legend.title = element_text(size=15))

# viz percentage
fimp %>% 
  ggplot(aes(x = percentage, 
             y = reorder(variable, -percentage))) +
  geom_point(color = "dark red", 
             fill = "dark red", 
             alpha = 0.5,
             size = 5) +
  labs(#title = "Percentage-based Feature Importance",
       #subtitle = "Deep Neural Network (2 hidden layers with 16 total neurons)",
       x = "Percentage",
       y = "Feature") +
  theme_minimal()+
  theme(axis.title = element_text(size=15),
        axis.text = element_text(size=17),
        legend.text = element_text(size=13),
        legend.title = element_text(size=15))

# Across both of these, deep features 12 and 13 are the most important by a wide margin (note, this is a pair we didn't plot in the earlyer chunk of code). So let's plot these against each other as they were most influential in reconstruction (greatest weights), suggesting they contribute most to the latent structure (and the best representation of the original input space). Substantively, this means we should expect to see the clearest separation between PID across these two deep features. Let's take a look at validation and training sets side by side (should be similar patterns)...

codings_val2 <- h2o.deepfeatures(object = autoencoder, 
                                data = validation, 
                                layer = 1) %>%
  as.data.frame() %>%
  mutate(democrat = as.factor(as.vector(validation[ , 36]))) # note, respecifying to be NOT an h20 object as before

# train plot
tr <- ggplot(codings_train, aes(x = DF.L1.C12, 
                               y = DF.L1.C13, 
                               color = factor(democrat))) +
  geom_point(alpha = 0.6) + 
  stat_ellipse() +
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Party",
                     breaks=c("0", "1"),
                     labels=c("Non-Democrat", "Democrat")) +
  labs(title = "Training Set",
       color = "Democrat") + 
  theme_minimal()+
  theme(axis.title = element_text(size=15),
        axis.text = element_text(size=17),
        legend.text = element_text(size=13),
        legend.title = element_text(size=15))

# valid plot
val <- ggplot(codings_val2, aes(x = DF.L1.C12, 
                          y = DF.L1.C13, 
                          color = factor(democrat))) +
  geom_point(alpha = 0.6) + 
  stat_ellipse() +
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Party",
                     breaks=c("0", "1"),
                     labels=c("Non-Democrat", "Democrat")) +
  labs(title = "Validation Set",
       color = "Democrat") + 
  theme_minimal()+
  theme(axis.title = element_text(size=15),
        axis.text = element_text(size=17),
        legend.text = element_text(size=13),
        legend.title = element_text(size=15))

# now viz
(tr + val)

# sure enough, we see really clear distinction between parties across these most important features. 

# Shut down h2o and stop the parallel cluster when finished with the session
h2o.shutdown()
stopCluster(cluster)
