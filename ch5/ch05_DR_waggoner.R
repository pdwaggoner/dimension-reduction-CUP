# Modern Dimension Reduction (Cambridge University Press)
# Philip D. Waggoner, pdwaggoner@uchicago.edu

# Section 5: Neural Network-Based Approaches


# Packages needed for this section:
install.packages(c("tidyverse", "here", "amerika", "tictoc", "kohonen", "h2o", "doParallel", "patchwork"))

# Load libraries
library(tidyverse)
library(here)
library(amerika)
library(tictoc)
library(kohonen)
library(patchwork)

# First, read in cleaned and preprocessed 2019 ANES Pilot Data
anes <- read_rds(here("Data", "anes.rds"))

set.seed(1234)

## SOM
anes_scaled <- anes[ ,1:35] %>% 
  scale()

search_grid <- somgrid(xdim = 10, 
                       ydim = 10, 
                       topo = "rectangular",
                       neighbourhood.fct = "gaussian") 

# train
{
  tic()
som_fit <- som(anes_scaled,
               grid = search_grid,
               alpha = c(0.1, 0.001),
               radius = 1,
               rlen = 500, 
               dist.fcts = "euclidean", 
               mode = "batch") 
  toc()
} # ~12.275 seconds

# plot training progress
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

# Classification assignments from SOM (and some compared to other clustering like k-means (hard part) and fuzzy c-means (soft part))
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

# Another viz
plot(som_fit, 
     type = "codes")

# codes by feature
# trump and obama (negative relationship)
som_fit$codes %>% 
  as.data.frame() %>% 
  ggplot(aes(Trump, Obama)) +
  geom_point() +
  geom_smooth(method = "loess", se = FALSE) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_vline(xintercept = 0, linetype = "dashed") +
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
  theme_minimal()+
  theme(axis.title = element_text(size=15),
        axis.text = element_text(size=17),
        legend.text = element_text(size=13),
        legend.title = element_text(size=15))

# UN and NRA (negative relationship)
som_fit$codes %>% 
  as.data.frame() %>% 
  ggplot(aes(UN, NRA)) +
  geom_point() +
  geom_smooth(method = "loess", se = FALSE) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_vline(xintercept = 0, linetype = "dashed") +
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
  theme_minimal()+
  theme(axis.title = element_text(size=15),
        axis.text = element_text(size=17),
        legend.text = element_text(size=13),
        legend.title = element_text(size=15))

#


## Autoencoders
library(h2o)
library(doParallel)

set.seed(1234)

anes$democrat <- factor(anes$democrat)

# initializing the h2o cluster
my_h2o <- h2o.init()

# parallel set up
cores <- detectCores() - 1 
cluster <- makeCluster(cores, setup_timeout = 0.5)
registerDoParallel(cluster) 

# df
anes_h2o <- anes %>% 
  as.h2o()

# train, test, validation
split_frame <- h2o.splitFrame(anes_h2o, 
                              ratios = c(0.6, 0.2), 
                              seed = 1234)   

split_frame %>% 
  glimpse()

train <- split_frame[[1]]
validation <- split_frame[[2]]
test <- split_frame[[3]]

# store some stuff
response <- "democrat"

predictors <- setdiff(colnames(train), response)

# vanilla AE with tanh activation and a single hidden layer with 16 nodes
{
  tic()
autoencoder <- h2o.deeplearning(x = predictors, 
                                training_frame = train,
                                autoencoder = TRUE,
                                reproducible = TRUE,
                                seed = 1234,
                                hidden = c(16),
                                epochs = 100,
                                activation = "Tanh",
                                validation_frame = test)
  toc()
} # ~5.3 seconds

# extract
codings_train <- h2o.deepfeatures(autoencoder, 
                                  data = train, 
                                  layer = 1) %>% 
  as.data.frame() %>%
  mutate(democrat = as.vector(train[ , 36]))


# viz
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

# new task
codings_val <- h2o.deepfeatures(object = autoencoder, 
                                data = validation, 
                                layer = 1) %>%
  as.data.frame() %>%
  mutate(democrat = as.factor(as.vector(validation[ , 36]))) %>%
  as.h2o()

# Store
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

# preds & classify
test_3 <- h2o.deepfeatures(object = autoencoder, 
                           data = test, 
                           layer = 1)

test_pred <- h2o.predict(deep_net, test_3, type = "response") %>%
  as.data.frame() %>%
  mutate(truth = as.vector(test[, 36]))

# conf mat
print(h2o.predict(deep_net, test_3) %>%
        as.data.frame() %>%
        mutate(truth = as.vector(test[, 36])) %>%
        group_by(truth, predict) %>%
        summarise(n = n()) %>%
        mutate(freq = n / sum(n)))

table(h2o.predict(deep_net, test_3))

## Feature importance
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
  labs(x = "Relative Importance",
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
  labs(x = "Percentage",
       y = "Feature") +
  theme_minimal()+
  theme(axis.title = element_text(size=15),
        axis.text = element_text(size=17),
        legend.text = element_text(size=13),
        legend.title = element_text(size=15))

# zoom in
codings_val2 <- h2o.deepfeatures(object = autoencoder, 
                                data = validation, 
                                layer = 1) %>%
  as.data.frame() %>%
  mutate(democrat = as.factor(as.vector(validation[ , 36]))) 

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

# now viz side by side
(tr + val)

# shut down cluster and h2o
h2o.shutdown()
stopCluster(cluster)
