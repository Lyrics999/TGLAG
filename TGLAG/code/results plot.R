library(readxl)
library(ggplot2)
library(tidyverse)
library(patchwork)


data = read_excel(".result.xlsx")
library(ggplot2)
font_family <- "serif"
data1=data[4]
data2=data[8]
df = data.frame(x = 1:316, y1 = data1, y2 = data2) 
colnames(df)[2] <- "y1"
colnames(df)[3] <- "y2"
font_family <- "serif"

ggplot(df,
       aes(x = x) 
)+
  geom_line(aes(y = y1, color ="predicted value")) + 
  geom_line(aes(y = y2, color = "actual value") )+ 
  labs(                         
    x = "Time(/day)",
    y = "temperature/(centigrade)",
    
  )+
  scale_color_manual(values = c("predicted value" = "red", "actual value"= "blue")) +
  theme_classic() +
  theme(
    legend.position = c(0.8, 0.8),   
    legend.title = element_blank(),
    #panel.border = element_rect(color = "black", fill = NA, size = 0.5),  #
    legend.text = element_text(size = 10),
    aspect.ratio = 1/2 ,
    text = element_text(family = font_family),
  ) 
