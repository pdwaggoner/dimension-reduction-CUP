# Nonparametric Dimension Reduction for High Dimensional Data (Cambridge University Press)
# Philip D. Waggoner, pdwaggoner@uchicago.edu

# Section 2: Starting with PCA


# Packages needed for this section:
install.packages(c("tidyverse", "here", "corrr", "amerika",
                   "factoextra", "patchwork", "ggrepel"))

# Load libraries
library(tidyverse)
library(here)
library(corrr)
library(amerika)
library(factoextra)
library(patchwork)
library(ggrepel)


# First, read in cleaned and preprocessed 2019 ANES Pilot Data
anes <- read_rds(here("Data", "anes.rds"))

# correlation and curse: correlation across all FTs
## fttrump
anes %>%
  select(-democrat) %>% 
  correlate(use = "pairwise.complete.obs",
            method = "pearson",
            quiet = TRUE) %>% 
  focus(Trump) %>%
  mutate(rowname = reorder(rowname, Trump)) %>%
  ggplot(aes(rowname, Trump)) +
  geom_col() + 
  coord_flip() + 
  labs(y = "Feelings Toward Trump", 
       x = "All Other Feeling Thermometers") +
  theme_minimal() +
  theme(axis.title = element_text(size=15),
        axis.text = element_text(size=17))

## ftjapan
anes %>%
  select(-democrat) %>% 
  correlate(use = "pairwise.complete.obs",
            method = "pearson",
            quiet = TRUE) %>% 
  focus(Japan) %>%
  mutate(rowname = reorder(rowname, Japan)) %>%
  ggplot(aes(rowname, Japan)) +
  geom_col() + 
  coord_flip() + 
  labs(y = "Feelings Toward Japan", 
       x = "All Other Feeling Thermometers") + 
  theme_minimal() +
  theme(axis.title = element_text(size=15),
        axis.text = element_text(size=17))

## network viz

anes %>%
  select(-democrat) %>% 
  correlate(use = "pairwise.complete.obs",
            method = "pearson",
            quiet = TRUE) %>% 
  network_plot(colors = c(amerika_palettes$Democrat[1], 
                          amerika_palettes$Republican[1]),
               curved = FALSE) 

# fit the model

pca_fit <- anes[,-36] %>%
  scale() %>% 
  prcomp(); summary(pca_fit)

# viz options

variance <- tibble(
  var = pca_fit$sdev^2,
  var_exp = var / sum(var),
  cum_var_exp = cumsum(var_exp)
) %>%
  mutate(pc = row_number())

# PVE
pve <- ggplot(variance, aes(pc, var_exp)) +
  geom_point() +
  geom_line() +
  geom_label_repel(aes(label = pc), size = 4) +
  labs(x = "Principal Component",
       y = "PVE") +
  theme_minimal() +
  theme(axis.title = element_text(size=15),
        axis.text = element_text(size=17))

# CPVE
cpve <- ggplot(variance, aes(pc, cum_var_exp)) +
  geom_point() +
  geom_line() +
  geom_label_repel(aes(label = pc), size = 4) +
  labs(x = "Principal Component",
       y = "Cumulative PVE") + 
  theme_minimal() +
  theme(axis.title = element_text(size=15),
        axis.text = element_text(size=17))

# viz side by side via patchwork
pve + cpve

# viz via factoextra
scree1 <- fviz_screeplot(pca_fit, main = "", addlabels = TRUE, choice = "variance")+
  theme(text = element_text(size=15),
        axis.text = element_text(size=17))
scree2 <- fviz_screeplot(pca_fit, main = "", addlabels = TRUE, choice = "eigenvalue")+
  theme(axis.title = element_text(size=15),
        axis.text = element_text(size=17))

scree1 + scree2 # side by side

# biplot
pca_fit %>% 
  fviz_pca_biplot(label = "var",
                  col.var = amerika_palettes$Republican[2],
                  col.ind = amerika_palettes$Democrat[3]) +
  labs(title = "") +
  theme_minimal()+
  theme(axis.title = element_text(size=15),
        axis.text = element_text(size=17))

# feature loadings/contributions ("contrib")
pca_fit %>% 
  fviz_pca_var(col.var = "contrib") +
  scale_color_gradient(high = amerika_palettes$Democrat[1], 
                       low = amerika_palettes$Republican[1]) +
  labs(color = "Contribution",
       title = "") +
  theme_minimal()+
  theme(axis.title = element_text(size=15),
        axis.text = element_text(size=17))

# custom, full viz
anes %>% 
  ggplot(aes(pca_fit$x[, 1], # `x` stores PC scores for all observations in the data
             pca_fit$x[, 2], 
             col = factor(democrat))) +
  geom_point() +
  stat_ellipse() +
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                      name="Party",
                      breaks=c("0", "1"),
                      labels=c("Non-Democrat", "Democrat")) +
  labs(x = "Principal Component 1",
       y = "Principal Component 2") +
  theme_minimal()+
  theme(axis.title = element_text(size=15),
        axis.text = element_text(size=17),
        legend.text = element_text(size=13),
        legend.title = element_text(size=15))
