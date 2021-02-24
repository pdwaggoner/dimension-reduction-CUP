# Modern Dimension Reduction (Cambridge University Press)
# Philip D. Waggoner, pdwaggoner@uchicago.edu

# Section 3: Locally linear embedding


# Packages needed for this section:
install.packages(c("tidyverse", "here", "lle", "plot3D",
                   "amerika", "parallel", "ggrepel", 
                   "tictoc", "patchwork"))

# Load libraries
library(tidyverse)
library(here)
library(lle)
library(plot3D)
library(amerika)
library(parallel)
library(ggrepel)
library(tictoc)
library(patchwork)

# First, read in cleaned and preprocessed 2019 ANES Pilot Data
anes <- read_rds(here("Data", "anes.rds"))

set.seed(1234)

# take a quick look at the data numerically and visually
skimr::skim(anes)

anes_scaled <- anes[, 1:35] %>% 
  scale() %>% 
  as_tibble() # for easier interface with plotting in a bit

# Some 3D viz to start
{
  par(mfrow = c(2,2))
  scatter3D(anes_scaled$Trump, 
            anes_scaled$Obama, 
            anes_scaled$Sanders,
            bty = "f",
            pch = 1,
            phi = 7,
            theta = 25,
            colkey = FALSE,
            col = ramp.col(c(amerika_palettes$Republican[1], 
                             amerika_palettes$Democrat[1])),
            main = "Politicians",
            xlab = "Donald Trump",
            ylab = "Barack Obama",
            zlab = "Bernie Sanders"
  )
  
  scatter3D(anes_scaled$NRA, 
            anes_scaled$NATO, 
            anes_scaled$UN,
            bty = "f",
            pch = 1,
            phi = 7,
            theta = 25,
            colkey = FALSE,
            col = ramp.col(c(amerika_palettes$Republican[1], 
                             amerika_palettes$Democrat[1])),
            main = "Institutions",
            xlab = "NRA",
            ylab = "NATO",
            zlab = "UN"
  )
  scatter3D(anes_scaled$Illegal, 
            anes_scaled$Immigrants, 
            anes_scaled$ICE,
            bty = "f",
            pch = 1,
            phi = 7,
            theta = 25,
            colkey = FALSE,
            col = ramp.col(c(amerika_palettes$Republican[1], 
                             amerika_palettes$Democrat[1])),
            main = "Issues (Immigration)",
            xlab = "Illegal Immigrants",
            ylab = "Immigrants",
            zlab = "ICE"
  )
  scatter3D(anes_scaled$Palestine, 
            anes_scaled$`Saudi Arabia`, 
            anes_scaled$Israel,
            bty = "f",
            pch = 1,
            phi = 7,
            theta = 25,
            colkey = FALSE,
            col = ramp.col(c(amerika_palettes$Republican[1], 
                             amerika_palettes$Democrat[1])),
            main = "Countries (Middle East)",
            xlab = "Palestine",
            ylab = "Saudi Arabia",
            zlab = "Israel"
  )
  par(mfrow = c(1,1))
  }

# First, find optimal k (lle's version of a grid search)
cores <- detectCores() - 1 

tic() 
find_k <- calc_k(anes_scaled,
                m = 2, 
                parallel = TRUE,
                cpus = cores) 
toc() # ~ 10.9 minutes on 3 cores; ~ 9.2 minutes on 7 cores

# inspect -- what is the optimal value for k? (a couple options...)
## option 1: manually by arranging
find_k %>% 
  arrange(rho) # looks like k = 19 is optimal

## option 2: extracting via which.min()
find_k[which.min(find_k$rho), ] 

# Regardless of the option, use dplyr's filter() to *extract* based on min \rho
optimal_k_rho <- find_k %>% 
  arrange(rho) %>% 
  filter(rho == min(.))


## viz
find_k %>% 
  arrange(rho) %>% 
  ggplot(aes(k, rho)) +
  geom_line() +
  geom_point(color = ifelse(find_k$k == min(find_k$k), 
                            "red", 
                            "black")) +
  geom_vline(xintercept = optimal_k_rho$k, 
             linetype = "dashed", 
             color = "red") +
  geom_label_repel(aes(label = k),
                   box.padding = unit(0.5, 'lines')) +
  labs(x = "Neighborhood Size (k)",
       y = expression(rho)) +
  theme_minimal()

# fit the LLE algorithm with the hyperparameters set, and embed on 2 dimensions 
{
tic() 
lle_fit <- lle(anes_scaled,
               m = 2,
               nnk = TRUE,
               k = 19)
toc() # ~ 1.5 minutes on 3 cores; ~ 1.4 minutes on 7 cores
}

# full LLE viz
lle_viz <- anes %>% 
  ggplot(aes(x = lle_fit$Y[,1], # scores for d1
             y = lle_fit$Y[,2], # scores for d2
             col = factor(democrat))) +
  geom_point() +
  stat_ellipse() +
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Party",
                     breaks=c("0", "1"),
                     labels=c("Non-Democrat", 
                              "Democrat")) +
  labs(x = "First Dimension",
       y = "Second Dimension",
       title = "LLE") + 
  theme_minimal()+
  theme(axis.title = element_text(size=15),
        axis.text = element_text(size=17),
        legend.text = element_text(size=13),
        legend.title = element_text(size=15))
lle_viz

# Compare PCA and LLE

pca_fit <- anes[, 1:35] %>% 
  scale() %>% 
  prcomp()

pca_viz <- anes %>% 
  ggplot(aes(pca_fit$x[, 1], 
             pca_fit$x[, 2], 
             col = factor(democrat))) +
  geom_point() +
  stat_ellipse() +
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Party",
                     breaks=c("0", "1"),
                     labels=c("Non-Democrat", 
                              "Democrat")) +
  labs(x = "Principal Component 1",
       y = "Principal Component 2",
       title = "PCA") +
  theme_minimal()+
  theme(axis.title = element_text(size=15),
        axis.text = element_text(size=17),
        legend.text = element_text(size=13),
        legend.title = element_text(size=15))


# viz side by side
library(patchwork)

lle_viz + pca_viz


# Compare with raw inputs

p1 <- anes %>% 
  ggplot(aes(Trump, Obama, 
             color = factor(democrat))) +
  geom_density_2d() + 
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Party",
                     breaks=c("0", "1"),
                     labels=c("Non-Democrat", 
                              "Democrat")) +
  labs(x = "Feelings Toward Trump",
       y = "Feelings Toward Obama") +
  theme_minimal()+
  theme(axis.title = element_text(size=15),
        axis.text = element_text(size=17),
        legend.text = element_text(size=13),
        legend.title = element_text(size=15))

p2 <- anes %>% 
  ggplot(aes(ICE, Illegal, 
             color = factor(democrat))) +
  geom_density_2d() + 
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Party",
                     breaks=c("0", "1"),
                     labels=c("Non-Democrat", 
                              "Democrat")) +
  labs(x = "Feelings Toward ICE",
       y = "Feelings Toward Illegal Immigrants") +
  theme_minimal()+
  theme(axis.title = element_text(size=15),
        axis.text = element_text(size=17),
        legend.text = element_text(size=13),
        legend.title = element_text(size=15))

p3 <- anes %>% 
  ggplot(aes(UN, NATO, 
             color = factor(democrat))) +
  geom_density_2d() + 
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Party",
                     breaks=c("0", "1"),
                     labels=c("Non-Democrat", 
                              "Democrat")) +
  labs(x = "Feelings Toward the United Nations",
       y = "Feelings Toward NATO") +
  theme_minimal()+
  theme(axis.title = element_text(size=15),
        axis.text = element_text(size=17),
        legend.text = element_text(size=13),
        legend.title = element_text(size=15))

p4 <- anes %>% 
  ggplot(aes(Palestine, Israel, 
             color = factor(democrat))) +
  geom_density_2d() + 
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Party",
                     breaks=c("0", "1"),
                     labels=c("Non-Democrat", 
                              "Democrat")) +
  labs(x = "Feelings Toward Palestine",
       y = "Feelings Toward Israel") +
  theme_minimal()+
  theme(axis.title = element_text(size=15),
        axis.text = element_text(size=17),
        legend.text = element_text(size=13),
        legend.title = element_text(size=15))

# viz together
(p1 + p2) /
  (p3 + p4)
  
