# plotting.R
# ----------
# ggplot2 figures for the report/poster: training/validation loss curves,
# per-class F1 and AUROC bar plots, and per-class AUROC / F1 / precision
# heatmaps comparing the ResNet, ViT and ResNet+ViViT fusion models.
# Reads loss_history.csv and final_validation_metrics.csv written by the
# training scripts (edit the paths / model names at the top of each section).

# Themes
####################################################################
textsize=10
axistextsize=8
legendtextsize=8
legendpointsize=2
pointsize=0.3
textcolor="#252525"
axiscolor="#252525"
linecolor="#252525"  
backgroundcol="white"
pointsizen=0.1
linesize=0.1

gridlwd=0.4
clustercolumns = F
clusterrows = F
showrowdend = F
showcolumndend = F
showheatmaplegend = F

mythemenull = theme_bw() + theme(
  axis.line=element_blank(),
  axis.text=element_blank(),
  
  axis.ticks=element_blank(),
  panel.border = element_blank(),
  panel.grid = element_blank())

mythemenull = theme_bw() + theme(
  axis.line=element_blank(),
  axis.text=element_blank(),
  
  axis.ticks=element_blank(),
  panel.border = element_blank(),
  panel.grid = element_blank())

mytheme=
  theme(text = element_text(size=textsize,color=textcolor),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_line(color="grey67",size=0.05,linetype = 2),
        panel.border = element_blank(),
        legend.title =  element_blank(),
        legend.position = "right",
        legend.key.width = unit(0.3,"line"),
        legend.key.height = unit(0.7,"line"),
        legend.text = element_text(margin = margin(0,0,0,0.1, "line")),
        axis.line = element_line(color = linecolor,size=linesize),
        axis.ticks = element_line(color = linecolor,size=linesize),
        axis.text =  element_text(size = textsize,color=textcolor),
        axis.title = element_text(size = textsize,color=textcolor),
        plot.title = element_text(size = textsize,color=textcolor,face="plain"),
        strip.background = element_rect(color = NA, fill = "grey91"),
        strip.text.x = element_text(size=textsize,color=textcolor,margin = margin(0.05,0,0.05,0, "line"))
  )
####################################################################
library(ggplot2)
library(tidyr)
library(dplyr)
####################################################################
### LOSS PLOT
model = "legacy9_epochs10_70_20_10_permsplit.v2"
loss_history <- read_csv(paste0("classes/cs231n/figures/2D_ResNet.v2/", model,"/loss_history.csv"))

loss_long <- loss_history %>% pivot_longer(cols = c(train_loss, val_loss),  names_to = "type",  values_to = "loss")
loss_long <- loss_long %>% mutate(type = recode(type, train_loss = "training", val_loss = "validation"))

p1 <- ggplot(loss_long, aes(x = epoch, y = loss, color = type)) +
  mytheme + theme_bw() +
  geom_line() +
  labs(title = "Training and Validation Loss",
       x = "Epoch",
       y = "Loss",
       color = NULL) +
  theme(
    legend.position = c(0.99, 0.99),
    legend.justification = c("right", "top"),
    legend.background = element_rect(fill = "white", color = "grey90"),
    legend.spacing.y = unit(0.1, "lines"),
    legend.key.height = unit(0.3, "lines"),
    plot.title = element_text(hjust = 0.5)  # Center the title
  )+scale_color_manual(values = c("training" = "#1f77b4", "validation" = "#ff7f0e"))

pdf(paste0("/labs/khatrilab/maestri/classes/cs231n/figures/2D_ResNet.v2/", model, "/loss_R.pdf"), height = 4.5, width = 8)
p1
dev.off()

####################################################################
### Per Class F1 Plot
classes <- c("Medical material",  "Arterial wall calcification", "Cardiomegaly",  "Pericardial effusion", "Coronary artery wall calcification",
             "Hiatal hernia", "Lymphadenopathy" , "Emphysema","Atelectasis" ,  "Lung nodule",  "Lung opacity" , "Pulmonary fibrotic sequela", 
             "Pleural effusion" , "Mosaic attenuation pattern" ,"Peribronchial thickening",  "Consolidation", "Bronchiectasis", "Interlobular septal thickening") 

# ResNet Results
model = "5_epochs_400px_100_slices_vlmclassone_resNet"
test_metrics <- read_csv(paste0("classes/cs231n/figures/vlm_challenge/", model,"/final_validation_metrics.csv"))
f1_values <- as.numeric(test_metrics$per_class_f1, strsplit(gsub("\\[|\\]", "", test_metrics$per_class_f1), ",\\s*")[[1]])
f1_string <- gsub("\\[|\\]", "", test_metrics$per_class_f1)
f1_values <- as.numeric(strsplit(f1_string, ",\\s*")[[1]])
vit_df <- data.frame(model = "ViT-class", perclass_f1= f1_values, classes = classes)


model = "5_epoch_400px_100_slices_video_ResNet"
test_metrics <- read_csv(paste0("classes/cs231n/figures/vlm_challenge/", model,"/final_validation_metrics.csv"))
f1_values <- as.numeric(test_metrics$per_class_f1, strsplit(gsub("\\[|\\]", "", test_metrics$per_class_f1), ",\\s*")[[1]])
f1_string <- gsub("\\[|\\]", "", test_metrics$per_class_f1)
f1_values <- as.numeric(strsplit(f1_string, ",\\s*")[[1]])
vit_df2 <- data.frame(model = "ViT-video", perclass_f1 = f1_values, classes = classes)

df_auc = rbind( vit_df,vit_df2)

p2 <- ggplot(df_auc, aes(x = classes, y = perclass_f1, fill = model)) +
    geom_bar(stat = "identity", position = "dodge") +
    theme_bw() +mytheme+
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ylab("Per Class F1")+theme(
    legend.key.height = unit(0.4, "cm"),
    legend.key.width = unit(0.4, "cm"),   
    legend.text = element_text(size = 10)  
  )+xlab("")+scale_fill_manual(values=cols.outcomes)

pdf(paste0("/labs/khatrilab/maestri/classes/cs231n/figures/2D_ResNet.v2/", model, "/perclass_f1.pdf"), height = 4.5, width = 8)
p2 + theme(
  plot.margin = margin(t = 10, r = 10, b = 10, l = 25)  
)
dev.off()

####################################################################
setwd("/labs/khatrilab/maestri")
library(readr)
library(ggplot2)
### Per Class AUC Plot
classes <- c("Medical material",  "Arterial wall calcification", "Cardiomegaly",  "Pericardial effusion", "Coronary artery wall calcification",
             "Hiatal hernia", "Lymphadenopathy" , "Emphysema","Atelectasis" ,  "Lung nodule",  "Lung opacity" , "Pulmonary fibrotic sequela", 
             "Pleural effusion" , "Mosaic attenuation pattern" ,"Peribronchial thickening",  "Consolidation", "Bronchiectasis", "Interlobular septal thickening") 

# ViT Results
model = "5_epochs_400px_100_slices_vlmclassone_resNet"
test_metrics <- read_csv(paste0("classes/cs231n/figures/vlm_challenge/", model,"/final_validation_metrics.csv"))
auc_values <- as.numeric(test_metrics$per_class_auc, strsplit(gsub("\\[|\\]", "", test_metrics$per_class_auc), ",\\s*")[[1]])
auc_values <- gsub("\\[|\\]", "", test_metrics$per_class_auc)
auc_values <- as.numeric(strsplit(auc_values, ",\\s*")[[1]])
vit_df <- data.frame(model = "ViT-class", perclass_auc_values = auc_values, classes = classes)

# Fusion Results
model = "5_epoch_400px_100_slices_video_ResNet"
test_metrics <- read_csv(paste0("classes/cs231n/figures/vlm_challenge/", model,"/final_validation_metrics.csv"))
auc_values <- as.numeric(test_metrics$per_class_auc, strsplit(gsub("\\[|\\]", "", test_metrics$per_class_auc), ",\\s*")[[1]])
auc_values <- gsub("\\[|\\]", "", test_metrics$per_class_auc)
auc_values <- as.numeric(strsplit(auc_values, ",\\s*")[[1]])
fusion_df <- data.frame(model = "ViT-video", perclass_auc_values = auc_values, classes = classes)


model = "5_epoch_400px_160_slices_video_ResNet"
test_metrics <- read_csv(paste0("classes/cs231n/figures/vlm_challenge/", model,"/final_validation_metrics.csv"))
auc_values <- as.numeric(test_metrics$per_class_auc, strsplit(gsub("\\[|\\]", "", test_metrics$per_class_auc), ",\\s*")[[1]])
auc_values <- gsub("\\[|\\]", "", test_metrics$per_class_auc)
auc_values <- as.numeric(strsplit(auc_values, ",\\s*")[[1]])
fusion_df2 <- data.frame(model = "ViT-video-160-slices", perclass_auc_values = auc_values, classes = classes)

model = "5_epoch_400px_100_slices_video_ResNet_modtransformer_params"
test_metrics <- read_csv(paste0("classes/cs231n/figures/vlm_challenge/", model,"/final_validation_metrics.csv"))
auc_values <- as.numeric(test_metrics$per_class_auc, strsplit(gsub("\\[|\\]", "", test_metrics$per_class_auc), ",\\s*")[[1]])
auc_values <- gsub("\\[|\\]", "", test_metrics$per_class_auc)
auc_values <- as.numeric(strsplit(auc_values, ",\\s*")[[1]])
vit_param<- data.frame(model = "ViT-video-params", perclass_auc_values = auc_values, classes = classes)

df_auc = rbind( vit_df,fusion_df,fusion_df2,vit_param)

cols.outcomes <- c("ViT-class"="#aecdce", "ViT-video"="#2e4a7d","ViT-video-160-slices"="lightgreen","ViT-video-params"="pink")
 

p2 <- ggplot(df_auc, aes(x = classes, y = perclass_auc_values, fill = model)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_bw() +mytheme+
  scale_y_continuous(
    expand = c(0, 0),
    limits = c(0, max(vit_df$perclass_auc_values) * 1.1)
  )+
  ylab("Per Class AUC")+theme(
    axis.title = element_text(size = 18),
    axis.text.x  = element_text(size = 18),
    axis.text.y  = element_text(size = 18),
    legend.key.height = unit(0.4, "cm"),  
    legend.key.width = unit(0.4, "cm"),  
    legend.text = element_text(size = 18),  
    legend.position = "top",          
    legend.direction = "horizontal",     
    legend.box = "horizontal",           
    legend.box.just = "left"          
    )+xlab("")+
  scale_fill_manual(values=cols.outcomes,breaks = c("ViT-class", "ViT-video","ViT-video-param","ViT-video-160-slices"))+
  coord_flip()



####################################################################
### Heatmap for poster
setwd("/labs/khatrilab/maestri")
library(readr)
library(tidyr)
library(dplyr)
library(ggplot2)
### Per Class AUC Plot
classes <- c("Medical material",  "Arterial wall calcification", "Cardiomegaly",  "Pericardial effusion", "Coronary artery wall calcification",
             "Hiatal hernia", "Lymphadenopathy" , "Emphysema","Atelectasis" ,  "Lung nodule",  "Lung opacity" , "Pulmonary fibrotic sequela", 
             "Pleural effusion" , "Mosaic attenuation pattern" ,"Peribronchial thickening",  "Consolidation", "Bronchiectasis", "Interlobular septal thickening") 

# ViT Results
model = "5_epochs_400px_100_slices_vlmclassone_resNet"
test_metrics <- read_csv(paste0("classes/cs231n/figures/vlm_challenge/", model,"/final_validation_metrics.csv"))
auc_values <- as.numeric(test_metrics$per_class_auc, strsplit(gsub("\\[|\\]", "", test_metrics$per_class_auc), ",\\s*")[[1]])
auc_values <- gsub("\\[|\\]", "", test_metrics$per_class_auc)
auc_values <- as.numeric(strsplit(auc_values, ",\\s*")[[1]])
vit_df <- data.frame(model = "ViT-class", perclass_auc_values = auc_values, classes = classes)

# Fusion Results
model = "5_epoch_400px_100_slices_video_ResNet"
test_metrics <- read_csv(paste0("classes/cs231n/figures/vlm_challenge/", model,"/final_validation_metrics.csv"))
auc_values <- as.numeric(test_metrics$per_class_auc, strsplit(gsub("\\[|\\]", "", test_metrics$per_class_auc), ",\\s*")[[1]])
auc_values <- gsub("\\[|\\]", "", test_metrics$per_class_auc)
auc_values <- as.numeric(strsplit(auc_values, ",\\s*")[[1]])
fusion_df <- data.frame(model = "ViT-video", perclass_auc_values = auc_values, classes = classes)

model = "5_epoch_400px_160_slices_video_ResNet"
test_metrics <- read_csv(paste0("classes/cs231n/figures/vlm_challenge/", model,"/final_validation_metrics.csv"))
auc_values <- as.numeric(test_metrics$per_class_auc, strsplit(gsub("\\[|\\]", "", test_metrics$per_class_auc), ",\\s*")[[1]])
auc_values <- gsub("\\[|\\]", "", test_metrics$per_class_auc)
auc_values <- as.numeric(strsplit(auc_values, ",\\s*")[[1]])
fusion_df2 <- data.frame(model = "ViT-video-160-slices", perclass_auc_values = auc_values, classes = classes)

model = "5_epoch_400px_160_slices_video_ResNet_NEW"
test_metrics <- read_csv(paste0("classes/cs231n/figures/vlm_challenge/", model,"/final_validation_metrics.csv"))
auc_values <- as.numeric(test_metrics$per_class_auc, strsplit(gsub("\\[|\\]", "", test_metrics$per_class_auc), ",\\s*")[[1]])
auc_values <- gsub("\\[|\\]", "", test_metrics$per_class_auc)
auc_values <- as.numeric(strsplit(auc_values, ",\\s*")[[1]])
fusion_dfNEW <- data.frame(model = "ViT-video-160-slices-NEW", perclass_auc_values = auc_values, classes = classes)

model = "5_epoch_400px_160_slices_video_ResNet_NEW_384dim"
test_metrics <- read_csv(paste0("classes/cs231n/figures/vlm_challenge/", model,"/final_validation_metrics.csv"))
auc_values <- as.numeric(test_metrics$per_class_auc, strsplit(gsub("\\[|\\]", "", test_metrics$per_class_auc), ",\\s*")[[1]])
auc_values <- gsub("\\[|\\]", "", test_metrics$per_class_auc)
auc_values <- as.numeric(strsplit(auc_values, ",\\s*")[[1]])
fusion_dfNEW2 <- data.frame(model = "ViT-video-160-slices-NEW-384dim", perclass_auc_values = auc_values, classes = classes)

model = "5_epoch_400px_100_slices_video_ResNet_modtransformer_params"
test_metrics <- read_csv(paste0("classes/cs231n/figures/vlm_challenge/", model,"/final_validation_metrics.csv"))
auc_values <- as.numeric(test_metrics$per_class_auc, strsplit(gsub("\\[|\\]", "", test_metrics$per_class_auc), ",\\s*")[[1]])
auc_values <- gsub("\\[|\\]", "", test_metrics$per_class_auc)
auc_values <- as.numeric(strsplit(auc_values, ",\\s*")[[1]])
vit_param<- data.frame(model = "ViT-video-params", perclass_auc_values = auc_values, classes = classes)

model = "5_epoch_new_loss"
test_metrics <- read_csv(paste0("classes/cs231n/figures/vlm_challenge/", model,"/final_validation_metrics.csv"))
auc_values <- as.numeric(test_metrics$per_class_auc, strsplit(gsub("\\[|\\]", "", test_metrics$per_class_auc), ",\\s*")[[1]])
auc_values <- gsub("\\[|\\]", "", test_metrics$per_class_auc)
auc_values <- as.numeric(strsplit(auc_values, ",\\s*")[[1]])
vit_param2<- data.frame(model = "ViT-new-loss", perclass_auc_values = auc_values, classes = classes)

df_auc = rbind( vit_df,fusion_df,fusion_df2,vit_param,vit_param2,fusion_dfNEW,fusion_dfNEW2 )

cols.outcomes <- c("ViT-video"="#aecdce", "ViT-class"="#2e4a7d","ViT-video-params"="pink","ViT-video-160-slices"="lightgreen","ViT-new-loss"="red","ViT-video-160-slices-NEW"="darkgreen",
                   "ViT-video-160-slices-NEW-384dim"="grey")
df_auc$model <- factor(df_auc$model, levels = c("ViT-video","ViT-class","ViT-video-params","ViT-video-160-slices","ViT-new-loss","ViT-video-160-slices-NEW","ViT-video-160-slices-NEW-384dim"))

heatmap_df <- df_auc %>%
  pivot_wider(names_from = model, values_from = perclass_auc_values) %>%
  arrange(desc(classes))

heatmap_long <- heatmap_df %>%
  pivot_longer(cols = c("ViT-video","ViT-class","ViT-video-params","ViT-video-160-slices","ViT-new-loss","ViT-video-160-slices-NEW",
                        "ViT-video-160-slices-NEW-384dim"), names_to = "model", values_to = "auc")

heatmap_long$model <- factor(heatmap_long$model, levels = rev(c("ViT-video", "ViT-class","ViT-video-params","ViT-video-160-slices","ViT-new-loss","ViT-video-160-slices-NEW",
                                                                "ViT-video-160-slices-NEW-384dim")))
heatmap_long$classes <- factor(heatmap_long$classes, levels = rev(classes))
mycol= c("#40A1FF", "#73DAFF", "#ABF8FF", "#E0FFFF", "#FFFFBF", "#FFE099", "#FFAD73", "#F76E5E", "#D92632","#A60021")

# heatmap
p2 <- ggplot(heatmap_long, aes(x = model, y = classes, fill = auc)) +
  geom_tile(color = "white", size = 0.5) +
  geom_text(aes(label = sprintf("%.3f", auc)), size = 4) +
  scale_fill_gradientn(colors = mycol, limits = c(0.45, 1)) +
  theme_minimal() +
  theme(
    axis.title = element_text(size = 14),
    axis.text.x = element_text(size = 16, angle = 45, hjust = 1),
    axis.text.y = element_text(size = 16),
    legend.key.height = unit(1, "cm"),
    legend.key.width = unit(0.5, "cm"),
    legend.title = element_text(size = 16),
    legend.text = element_text(size = 16),
    legend.position = "right",
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  ) + labs(x = "", y = "", fill = "AUC")
p2

p2+ 
theme(
 plot.margin = margin(t = 3, r = 0, b = -10, l = 0)
)

####################################################################
# Per class F1 heatmap

### Per Class F1 Plot
classes <- c("Medical material",  "Arterial wall calcification", "Cardiomegaly",  "Pericardial effusion", "Coronary artery wall calcification",
             "Hiatal hernia", "Lymphadenopathy" , "Emphysema","Atelectasis" ,  "Lung nodule",  "Lung opacity" , "Pulmonary fibrotic sequela", 
             "Pleural effusion" , "Mosaic attenuation pattern" ,"Peribronchial thickening",  "Consolidation", "Bronchiectasis", "Interlobular septal thickening") 

# ResNet Results
model = "5_epochs_400px_100_slices_vlmclassone_resNet"
test_metrics <- read_csv(paste0("classes/cs231n/figures/vlm_challenge/", model,"/final_validation_metrics.csv"))
f1_values <- as.numeric(test_metrics$per_class_f1, strsplit(gsub("\\[|\\]", "", test_metrics$per_class_f1), ",\\s*")[[1]])
f1_string <- gsub("\\[|\\]", "", test_metrics$per_class_f1)
f1_values <- as.numeric(strsplit(f1_string, ",\\s*")[[1]])
vit_df <- data.frame(model = "ViT-class", perclass_f1= f1_values, classes = classes)


model = "5_epoch_400px_100_slices_video_ResNet"
test_metrics <- read_csv(paste0("classes/cs231n/figures/vlm_challenge/", model,"/final_validation_metrics.csv"))
f1_values <- as.numeric(test_metrics$per_class_f1, strsplit(gsub("\\[|\\]", "", test_metrics$per_class_f1), ",\\s*")[[1]])
f1_string <- gsub("\\[|\\]", "", test_metrics$per_class_f1)
f1_values <- as.numeric(strsplit(f1_string, ",\\s*")[[1]])
vit_df2 <- data.frame(model = "ViT-video", perclass_f1 = f1_values, classes = classes)

model = "5_epoch_400px_160_slices_video_ResNet"
test_metrics <- read_csv(paste0("classes/cs231n/figures/vlm_challenge/", model,"/final_validation_metrics.csv"))
f1_values <- as.numeric(test_metrics$per_class_f1, strsplit(gsub("\\[|\\]", "", test_metrics$per_class_f1), ",\\s*")[[1]])
f1_string <- gsub("\\[|\\]", "", test_metrics$per_class_f1)
f1_values <- as.numeric(strsplit(f1_string, ",\\s*")[[1]])
vit_df2.5 <- data.frame(model = "ViT-video-160-slices", perclass_f1 = f1_values, classes = classes)

model = "5_epoch_400px_160_slices_video_ResNet_NEW"
test_metrics <- read_csv(paste0("classes/cs231n/figures/vlm_challenge/", model,"/final_validation_metrics.csv"))
f1_values <- as.numeric(test_metrics$per_class_f1, strsplit(gsub("\\[|\\]", "", test_metrics$per_class_f1), ",\\s*")[[1]])
f1_string <- gsub("\\[|\\]", "", test_metrics$per_class_f1)
f1_values <- as.numeric(strsplit(f1_string, ",\\s*")[[1]])
vit_df2.6 <- data.frame(model = "ViT-video-160-slices-NEW", perclass_f1 = f1_values, classes = classes)

model = "5_epoch_400px_160_slices_video_ResNet_NEW_384dim"
test_metrics <- read_csv(paste0("classes/cs231n/figures/vlm_challenge/", model,"/final_validation_metrics.csv"))
f1_values <- as.numeric(test_metrics$per_class_f1, strsplit(gsub("\\[|\\]", "", test_metrics$per_class_f1), ",\\s*")[[1]])
f1_string <- gsub("\\[|\\]", "", test_metrics$per_class_f1)
f1_values <- as.numeric(strsplit(f1_string, ",\\s*")[[1]])
vit_df2.7 <- data.frame(model = "ViT-video-160-slices-NEW-384dim", perclass_f1 = f1_values, classes = classes)


model = "5_epoch_400px_100_slices_video_ResNet_modtransformer_params"
test_metrics <- read_csv(paste0("classes/cs231n/figures/vlm_challenge/", model,"/final_validation_metrics.csv"))
f1_values <- as.numeric(test_metrics$per_class_f1, strsplit(gsub("\\[|\\]", "", test_metrics$per_class_f1), ",\\s*")[[1]])
f1_string <- gsub("\\[|\\]", "", test_metrics$per_class_f1)
f1_values <- as.numeric(strsplit(f1_string, ",\\s*")[[1]])
vit_df3 <- data.frame(model = "ViT-video-params", perclass_f1 = f1_values, classes = classes)


model = "5_epoch_new_loss"
test_metrics <- read_csv(paste0("classes/cs231n/figures/vlm_challenge/", model,"/final_validation_metrics.csv"))
f1_values <- as.numeric(test_metrics$per_class_f1, strsplit(gsub("\\[|\\]", "", test_metrics$per_class_f1), ",\\s*")[[1]])
f1_string <- gsub("\\[|\\]", "", test_metrics$per_class_f1)
f1_values <- as.numeric(strsplit(f1_string, ",\\s*")[[1]])
vit_df4 <- data.frame(model = "ViT-new-loss", perclass_f1 = f1_values, classes = classes)

df_auc = rbind( vit_df,vit_df2,vit_df2.5, vit_df3,vit_df4,vit_df2.6, vit_df2.7)

cols.outcomes <- c("ViT-video"="#aecdce", "ViT-class"="#2e4a7d","ViT-video-params"="pink","ViT-video-160-slices"="lightgreen","ViT-new-loss"="red","ViT-video-160-slices-NEW"="darkgreen",
                   "ViT-video-160-slices-NEW-384dim"="grey")
df_auc$model <- factor(df_auc$model, levels = c("ViT-video","ViT-class","ViT-video-params","ViT-video-160-slices","ViT-new-loss","ViT-video-160-slices-NEW","ViT-video-160-slices-NEW-384dim"))

heatmap_df <- df_auc %>%
  pivot_wider(names_from = model, values_from = perclass_f1) %>%
  arrange(desc(classes))

heatmap_long <- heatmap_df %>%
  pivot_longer(cols = c("ViT-video","ViT-class","ViT-video-params","ViT-video-160-slices","ViT-new-loss","ViT-video-160-slices-NEW",
                        "ViT-video-160-slices-NEW-384dim"), names_to = "model", values_to = "f1")

heatmap_long$model <- factor(heatmap_long$model, levels = rev(c("ViT-video", "ViT-class","ViT-video-params","ViT-video-160-slices","ViT-new-loss","ViT-video-160-slices-NEW",
                                                                "ViT-video-160-slices-NEW-384dim")))
heatmap_long$classes <- factor(heatmap_long$classes, levels = rev(classes))
mycol= c("#40A1FF", "#73DAFF", "#ABF8FF", "#E0FFFF", "#FFFFBF", "#FFE099", "#FFAD73", "#F76E5E", "#D92632","#A60021")

# heatmap
p2 <- ggplot(heatmap_long, aes(x = model, y = classes, fill = f1)) +
  geom_tile(color = "white", size = 0.5) +
  geom_text(aes(label = sprintf("%.3f", f1)), size = 4) +
  scale_fill_gradientn(colors = mycol, limits = c(0.45, 1)) +
  theme_minimal() +
  theme(
    axis.title = element_text(size = 14),
    axis.text.x = element_text(size = 16, angle = 45, hjust = 1),
    axis.text.y = element_text(size = 16),
    legend.key.height = unit(1, "cm"),
    legend.key.width = unit(0.5, "cm"),
    legend.title = element_text(size = 16),
    legend.text = element_text(size = 16),
    legend.position = "right",
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  ) + labs(x = "", y = "", fill = "f1")

p2


####
# Precision heatmap

classes <- c("Medical material",  "Arterial wall calcification", "Cardiomegaly",  "Pericardial effusion", "Coronary artery wall calcification",
             "Hiatal hernia", "Lymphadenopathy" , "Emphysema","Atelectasis" ,  "Lung nodule",  "Lung opacity" , "Pulmonary fibrotic sequela", 
             "Pleural effusion" , "Mosaic attenuation pattern" ,"Peribronchial thickening",  "Consolidation", "Bronchiectasis", "Interlobular septal thickening") 

# ResNet Results
model = "5_epochs_400px_100_slices_vlmclassone_resNet"
test_metrics <- read_csv(paste0("classes/cs231n/figures/vlm_challenge/", model,"/final_validation_metrics.csv"))
f1_values <- as.numeric(test_metrics$per_class_precision, strsplit(gsub("\\[|\\]", "", test_metrics$per_class_precision), ",\\s*")[[1]])
f1_string <- gsub("\\[|\\]", "", test_metrics$per_class_precision)
f1_values <- as.numeric(strsplit(f1_string, ",\\s*")[[1]])
vit_df <- data.frame(model = "ViT-class", per_class_precision= f1_values, classes = classes)


model = "5_epoch_400px_100_slices_video_ResNet"
test_metrics <- read_csv(paste0("classes/cs231n/figures/vlm_challenge/", model,"/final_validation_metrics.csv"))
f1_values <- as.numeric(test_metrics$per_class_precision, strsplit(gsub("\\[|\\]", "", test_metrics$per_class_precision), ",\\s*")[[1]])
f1_string <- gsub("\\[|\\]", "", test_metrics$per_class_precision)
f1_values <- as.numeric(strsplit(f1_string, ",\\s*")[[1]])
vit_df2 <- data.frame(model = "ViT-video", per_class_precision = f1_values, classes = classes)

model = "5_epoch_400px_160_slices_video_ResNet"
test_metrics <- read_csv(paste0("classes/cs231n/figures/vlm_challenge/", model,"/final_validation_metrics.csv"))
f1_values <- as.numeric(test_metrics$per_class_precision, strsplit(gsub("\\[|\\]", "", test_metrics$per_class_precision), ",\\s*")[[1]])
f1_string <- gsub("\\[|\\]", "", test_metrics$per_class_precision)
f1_values <- as.numeric(strsplit(f1_string, ",\\s*")[[1]])
vit_df2.5 <- data.frame(model = "ViT-video-160-slices", per_class_precision = f1_values, classes = classes)


model = "5_epoch_400px_100_slices_video_ResNet_modtransformer_params"
test_metrics <- read_csv(paste0("classes/cs231n/figures/vlm_challenge/", model,"/final_validation_metrics.csv"))
f1_values <- as.numeric(test_metrics$per_class_precision, strsplit(gsub("\\[|\\]", "", test_metrics$per_class_precision), ",\\s*")[[1]])
f1_string <- gsub("\\[|\\]", "", test_metrics$per_class_precision)
f1_values <- as.numeric(strsplit(f1_string, ",\\s*")[[1]])
vit_df3 <- data.frame(model = "ViT-video-params", per_class_precision = f1_values, classes = classes)

df_auc = rbind( vit_df,vit_df2,vit_df2.5, vit_df3)

cols.outcomes <- c("ViT-video"="#aecdce", "ViT-class"="#2e4a7d","ViT-video-params"="pink","ViT-video-160-slices"="lightgreen")
df_auc$model <- factor(df_auc$model, levels = c("ViT-video","ViT-class","ViT-video-params","ViT-video-160-slices"))

heatmap_df <- df_auc %>%
  pivot_wider(names_from = model, values_from = per_class_precision) %>%
  arrange(desc(classes))

heatmap_long <- heatmap_df %>%
  pivot_longer(cols = c("ViT-video","ViT-class","ViT-video-params","ViT-video-160-slices"), names_to = "model", values_to = "per_class_precision")

heatmap_long$model <- factor(heatmap_long$model, levels = rev(c("ViT-video", "ViT-class","ViT-video-params","ViT-video-160-slices")))
heatmap_long$classes <- factor(heatmap_long$classes, levels = rev(classes))
mycol= c("#40A1FF", "#73DAFF", "#ABF8FF", "#E0FFFF", "#FFFFBF", "#FFE099", "#FFAD73", "#F76E5E", "#D92632","#A60021")

# heatmap
p2 <- ggplot(heatmap_long, aes(x = model, y = classes, fill = per_class_precision)) +
  geom_tile(color = "white", size = 0.5) +
  geom_text(aes(label = sprintf("%.3f", per_class_precision)), size = 4) +
  scale_fill_gradientn(colors = mycol, limits = c(0.45, 1)) +
  theme_minimal() +
  theme(
    axis.title = element_text(size = 14),
    axis.text.x = element_text(size = 16, angle = 45, hjust = 1),
    axis.text.y = element_text(size = 16),
    legend.key.height = unit(1, "cm"),
    legend.key.width = unit(0.5, "cm"),
    legend.title = element_text(size = 16),
    legend.text = element_text(size = 16),
    legend.position = "right",
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  ) + labs(x = "", y = "", fill = "f1")

p2
