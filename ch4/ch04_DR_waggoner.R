# Modern Dimension Reduction (Cambridge University Press)
# Philip D. Waggoner, pdwaggoner@uchicago.edu

# Section 4: t-SNE and UMAP


# Packages needed for this section:
install.packages(c("tidyverse", "here", "amerika", "tictoc",
                   "patchwork", "Rtsne", "umap", "tidymodels",
                   "embed"))

# Load libraries
library(tidyverse)
library(here)
library(amerika)
library(tictoc)
library(patchwork)
library(Rtsne)
library(umap)
library(tidymodels)
library(embed)

# First, read in cleaned and preprocessed 2019 ANES Pilot Data
anes <- read_rds(here("Data", "anes.rds"))

# t-SNE manual

set.seed(1234)

{
  tic()
  
  # perplexity = 2
  tsne_2 <- Rtsne(as.matrix(anes[ ,1:35]), 
                  perplexity = 2)
  
  perp_2 <- anes %>%
    ggplot(aes(tsne_2$Y[,1], tsne_2$Y[,2], 
               col = factor(democrat))) +
    geom_point() +
    stat_ellipse() +
    scale_color_manual(values=c(amerika_palettes$Republican[1], 
                                amerika_palettes$Democrat[1]),
                       name="Democrat",
                       breaks=c("0", "1"),
                       labels=c("No", 
                                "Yes")) +
    ylim(-100, 100) +
    xlim(-100, 100) +
    labs(x = "First dimension",
         y = "Second dimension",
         subtitle = "Perplexity = 2") +
    theme_minimal()
  
  
  # perplexity = 5
  tsne_5 <- Rtsne(as.matrix(anes[ ,1:35]), 
                  perplexity = 5) 
  
  perp_5 <- anes %>%
    ggplot(aes(tsne_5$Y[,1], tsne_5$Y[,2], 
               col = factor(democrat))) +
    geom_point() +
    stat_ellipse() +
    scale_color_manual(values=c(amerika_palettes$Republican[1], 
                                amerika_palettes$Democrat[1]),
                       name="Democrat",
                       breaks=c("0", "1"),
                       labels=c("No", 
                                "Yes")) +
    ylim(-100, 100) +
    xlim(-100, 100) +
    labs(x = "First dimension",
         y = "Second dimension",
         subtitle = "Perplexity = 5") +
    theme_minimal()
  
  
  # perplexity = 25
  tsne_25 <- Rtsne(as.matrix(anes[ ,1:35]), 
                   perplexity = 25) 
  
  perp_25 <- anes %>%
    ggplot(aes(tsne_25$Y[,1], tsne_25$Y[,2], 
               col = factor(democrat))) +
    geom_point() +
    stat_ellipse() +
    scale_color_manual(values=c(amerika_palettes$Republican[1], 
                                amerika_palettes$Democrat[1]),
                       name="Democrat",
                       breaks=c("0", "1"),
                       labels=c("No", 
                                "Yes")) +
    ylim(-100, 100) +
    xlim(-100, 100) +
    labs(x = "First dimension",
         y = "Second dimension",
         subtitle = "Perplexity = 25") +
    theme_minimal()
  
  
  # perplexity = 50
  tsne_50 <- Rtsne(as.matrix(anes[ ,1:35]), 
                   perplexity = 50) 
  
  perp_50 <- anes %>%
    ggplot(aes(tsne_50$Y[,1], tsne_50$Y[,2], 
               col = factor(democrat))) +
    geom_point() +
    stat_ellipse() +
    scale_color_manual(values=c(amerika_palettes$Republican[1], 
                                amerika_palettes$Democrat[1]),
                       name="Democrat",
                       breaks=c("0", "1"),
                       labels=c("No", 
                                "Yes")) +
    ylim(-100, 100) +
    xlim(-100, 100) +
    labs(x = "First dimension",
         y = "Second dimension",
         subtitle = "Perplexity = 50") +
    theme_minimal()
  
  
  # perplexity = 100
  tsne_100 <- Rtsne(as.matrix(anes[ ,1:35]), 
                    perplexity = 100) 
  
  perp_100 <- anes %>%
    ggplot(aes(tsne_100$Y[,1], tsne_100$Y[,2], 
               col = factor(democrat))) +
    geom_point() +
    stat_ellipse() +
    scale_color_manual(values=c(amerika_palettes$Republican[1], 
                                amerika_palettes$Democrat[1]),
                       name="Democrat",
                       breaks=c("0", "1"),
                       labels=c("No", 
                                "Yes")) +
    ylim(-100, 100) +
    xlim(-100, 100) +
    labs(x = "First dimension",
         y = "Second dimension",
         subtitle = "Perplexity = 100") +
    theme_minimal()
  
  
  # perplexity = 500
  tsne_500 <- Rtsne(as.matrix(anes[ ,1:35]), 
                    perplexity = 500) 
  
  perp_500 <- anes %>%
    ggplot(aes(tsne_500$Y[,1], tsne_500$Y[,2], 
               col = factor(democrat))) +
    geom_point() +
    stat_ellipse() +
    scale_color_manual(values=c(amerika_palettes$Republican[1], 
                                amerika_palettes$Democrat[1]),
                       name="Democrat",
                       breaks=c("0", "1"),
                       labels=c("No", 
                                "Yes")) +
    ylim(-100, 100) +
    xlim(-100, 100) +
    labs(x = "First dimension",
         y = "Second dimension",
         subtitle = "Perplexity = 500") +
    theme_minimal()
  
  toc()
} # ~1.6 minutes


# Visualize
tsne_plots <- (perp_2 + perp_5 + perp_25) /
  (perp_50 + perp_100 + perp_500)

tsne_plots

## with annotation if desired
#tsne_plots + plot_annotation(title = "t-SNE Results Across a Range of Perplexity",
#                             subtitle = "Color conditional on Party Affiliation")


# t-SNE grid search

{
  hyperparameters_tsne <- expand.grid(perplexity = c(2, 5, 25, 50, 100, 500), 
                                      theta = c(0.0, 0.20, 0.40, 0.60, 0.80, 1.0)) # 0.0 for exact (e.g., X == Y), 0.5 is default
  tic()
  tsne_full <- pmap(hyperparameters_tsne, 
                    Rtsne, 
                    X = as.matrix(anes[ ,1:35]), 
                    dims = 2,
                    max_iter = 1000,
                    momentum = 0.5, 
                    final_momentum = 0.8, 
                    eta = 200)
  toc()
} # ~19 minutes to run

grid_values_tsne <- tibble(perplexity = rep(hyperparameters_tsne$perplexity, each = 3165),
                           theta = rep(hyperparameters_tsne$theta, each = 3165),
                           d1 = unlist(map(tsne_full, ~ .$Y[, 1])),
                           d2 = unlist(map(tsne_full, ~ .$Y[, 2])))

# viz
grid_values_tsne %>% 
  ggplot(aes(d1, d2)) +
  facet_grid(theta ~ perplexity) +
  geom_point() +
  labs(subtitle = "Theta x Perplexity",
       x = "First Dimension",
       y = "Second Dimension",
       caption = "Rows = Theta\nColumns = Perplexity") +
  theme_minimal()


#
# UMAP
#

# base UMAP with k = 5 and epochs = 500 (see the data a lot and thus learn better), all else set to default
{
  suppressWarnings(
umap_fit_5 <- anes[,1:35] %>% 
  umap(n_neighbors = 5,
       metric = "euclidean",
       n_epochs = 500)
)

  suppressWarnings(
umap_fit_5 <- anes %>% 
  mutate_if(.funs = scale,
            .predicate = is.numeric,
            scale = FALSE) %>% 
  mutate(First_Dimension = umap_fit_5$layout[,1],
         Second_Dimension = umap_fit_5$layout[,2]) %>% 
  gather(key = "Variable",
         value = "Value",
         c(-First_Dimension, -Second_Dimension, -democrat))
)
}

k_5 <- ggplot(umap_fit_5, aes(First_Dimension, Second_Dimension, 
                     col = factor(democrat))) + 
  geom_point(alpha = 0.6) +
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Democrat",
                     breaks=c("-0.418325434439179", 
                              "0.581674565560822"),
                     labels=c("No", 
                              "Yes")) +
  labs(title = " ",
       subtitle = "Neighborhood size: 5; Epochs = 500",
       x = "First Dimension",
       y = "Second Dimension") +
  theme_minimal()+
  theme(text = element_text(size=15),
        axis.text = element_text(size=17))


{
  suppressWarnings(
umap_fit_e_20 <- anes[,1:35] %>% 
  umap(n_neighbors = 5,
       metric = "euclidean",
       n_epochs = 20)
)
  suppressWarnings(
umap_fit_e_20 <- anes %>% 
  mutate_if(.funs = scale,
            .predicate = is.numeric,
            scale = FALSE) %>% 
  mutate(First_Dimension = umap_fit_e_20$layout[,1],
         Second_Dimension = umap_fit_e_20$layout[,2]) %>% 
  gather(key = "Variable",
         value = "Value",
         c(-First_Dimension, -Second_Dimension, -democrat))
)
  }

e_20 <- ggplot(umap_fit_e_20, aes(First_Dimension, Second_Dimension, 
                            col = factor(democrat))) + 
  geom_point(alpha = 0.6) +
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Democrat",
                     breaks=c("-0.418325434439179", 
                              "0.581674565560822"),
                     labels=c("No", 
                              "Yes")) +
  labs(title = " ",
       subtitle = "Neighborhood size: 5; Epochs = 20",
       x = "First Dimension",
       y = "Second Dimension") +
  theme_minimal()+
  theme(text = element_text(size=15),
        axis.text = element_text(size=17))

# side by side

suppressWarnings(
  k_5 + e_20
)

# UMAP grid search

{
hyperparameters_umap <- expand.grid(n_neighbors = seq(5, 50, 10),
                                    n_epochs    = seq(50, 450, 100))
  tic()
umap_full <- pmap(hyperparameters_umap, 
                  umap, 
                  d = anes[,1:35])
  toc()
} # ~5.5 minutes

grid_values_umap <- tibble(n_neighbors = rep(hyperparameters_umap$n_neighbors, each = 3165),
                           n_epochs = rep(hyperparameters_umap$n_epochs, each = 3165),
                           d1 = unlist(map(umap_full, ~ .$layout[, 1])),
                           d2 = unlist(map(umap_full, ~ .$layout[, 2])))

grid_values_umap %>%
  ggplot(aes(d1, d2)) +
  geom_point() +
  facet_grid(n_neighbors ~ n_epochs,
             scales = "fixed") +
  labs(x = "First Dimension",
       y = "Second Dimension",
       caption = "Rows = Neighborhood Sizes\nColumns = Epochs") +
  theme_minimal()


# Tidy approach

recipe(~ ., data = anes) %>%
  update_role(democrat, new_role = "id") %>%
  step_umap(all_predictors(),
            neighbors = 5,
            epochs = 500) %>% 
  prep() %>% 
  juice() %>% 
  ggplot(aes(umap_1, umap_2,
             color = factor(democrat))) +
  geom_point(alpha = 0.6) +
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Democrat",
                     breaks=c("0", "1"),
                     labels=c("No", 
                              "Yes")) +
  labs(title = " ",
       caption = "UMAP via tidymodels + uwot + embed",
       x = "First Dimension",
       y = "Second Dimension") +
  theme_minimal()+
  theme(axis.title = element_text(size=15),
        axis.text = element_text(size=17),
        legend.text = element_text(size=13),
        legend.title = element_text(size=15))
