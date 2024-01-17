
library(readxl)
library(ggplot2)
library(tidyverse)
library(patchwork)
library("magrittr")

dat01 = read_excel(".mae-Ablation experiment.xlsx")
dim(dat01)
dat01$mae

font_family <- "serif"
dat01$station<-factor(dat01$station,
                      levels = c("Kunshan","Nantong","Gaoyou","Suining"))
dat01$model <- factor(dat01$model, levels = c("GRU", "T-GCN", "T-GCN-Luong Attention","Combine Model"))
ggplot(data=dat01,aes(x=station,y=mae,fill=model))+
  geom_bar(stat="identity",position = "dodge")+
  theme_classic()+
  geom_vline(xintercept = 5.5,lty="dashed")+
  geom_vline(xintercept = 9.5,lty="dashed")+
  theme(legend.position = "bottom",
        axis.text.x = element_text(angle=0,hjust = 0.5,vjust = 0.5,size=11),
        legend.title = element_blank(),
        aspect.ratio = 1.5/2,
        text = element_text(family = font_family) )+
  labs(x=NULL,y="MAE/(centigrade)")+
  scale_fill_manual(values = c("#936eaa",
                                        "#5f6798","#de6eaa","#8DA0CB"))+
                                          scale_y_continuous(labels = function(x){x*100}) -> p1

p1



dat01 = read_excel(".rmsemae-Ablation experiment.xlsx")
dim(dat01)
dat01$rmse

font_family <- "serif"
dat01$station<-factor(dat01$station,
                      levels = c("Kunshan","Nantong","Gaoyou","Suining"))
dat01$model <- factor(dat01$model, levels = c("GRU", "T-GCN", "T-GCN-Luong Attention","Combine Model"))
ggplot(data=dat01,aes(x=station,y=rmse,fill=model))+
  geom_bar(stat="identity",position = "dodge")+
  theme_classic()+
  geom_vline(xintercept = 5.5,lty="dashed")+
  geom_vline(xintercept = 9.5,lty="dashed")+
  theme(legend.position = "bottom",
        axis.text.x = element_text(angle=0,hjust = 0.5,vjust = 0.5,size=11),
        legend.title = element_blank(),
        aspect.ratio = 1.5/2,
        text = element_text(family = font_family) )+
  labs(x=NULL,y="RMSE/(centigrade)")+
  scale_fill_manual(values = c("#936eaa",
                                        "#5f6798","#de6eaa","#8DA0CB"))+
                                          scale_y_continuous(labels = function(x){x*100}) -> p1

p1

