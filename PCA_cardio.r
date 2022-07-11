# dimension reduction

# Introduction
# World Health Organization has estimated 12 million deaths occur worldwide, every year due to Heart diseases. Half the deaths in the United States and other developed countries are due to cardio vascular diseases. The early prognosis of cardiovascular diseases can aid in making decisions on lifestyle changes in high risk patients and in turn reduce the complications. This research intends to pinpoint the most relevant/risk factors of heart disease as well as predict the overall risk using logistic regression
# Data Preparation
# 
# Source
# The dataset is publically available on the Kaggle website, and it is from an ongoing cardiovascular study on residents of the town of Framingham, Massachusetts. The classification goal is to predict whether the patient has 10-year risk of future coronary heart disease (CHD).The dataset provides the patientsí information. It includes over 4,000 records and 15 attributes.
# Variables
# Each attribute is a potential risk factor. There are both demographic, behavioral and medical risk factors.
# 
# Demographic:
#   ï Sex: male or female(Nominal)
# ï Age: Age of the patient;(Continuous - Although the recorded ages have been truncated to whole numbers, the concept of age is continuous)
# Behavioral
# ï Current Smoker: whether or not the patient is a current smoker (Nominal)
# ï Cigs Per Day: the number of cigarettes that the person smoked on average in one day.(can be considered continuous as one can have any number of cigarettes, even half a cigarette.)
# Medical( history)
# ï BP Meds: whether or not the patient was on blood pressure medication (Nominal)
# ï Prevalent Stroke: whether or not the patient had previously had a stroke (Nominal)
# ï Prevalent Hyp: whether or not the patient was hypertensive (Nominal)
# ï Diabetes: whether or not the patient had diabetes (Nominal)
# Medical(current)
# ï Tot Chol: total cholesterol level (Continuous)
# ï Sys BP: systolic blood pressure (Continuous)
# ï Dia BP: diastolic blood pressure (Continuous)
# ï BMI: Body Mass Index (Continuous)
# ï Heart Rate: heart rate (Continuous - In medical research, variables such as heart rate though in fact discrete, yet are considered continuous because of large number of possible values.)
# ï Glucose: glucose level (Continuous)
# Predict variable (desired target)
# ï 10 year risk of coronary heart disease CHD (binary: ì1î, means ìYesî, ì0î means ìNoî)


library(dplyr)

cardio<-read.csv2("oldml.csv", sep=",")
colnames(cardio)
cardio <- select(cardio, -c("male","education", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes", "TenYearCHD"))
cardio[,] <- apply(cardio[,], 2, function(x) as.numeric(as.character(x)))
cardio <- na.omit(cardio)

summary(cardio)
dim(cardio)

library(caret)

preproc1 <- preProcess(cardio, method=c("center", "scale"))
cardio.s <- predict(preproc1, cardio)
 
summary(cardio.s)

# PLOTS
# 1st plot
library(GGally)
ggpairs(cardio)
# 2nd plot
library(corrplot)
corr_df = cor(cardio, method='pearson')
corrplot(corr_df)


# eigenvalues on the basis of covariance
cardio.cov<-cov(cardio.s)
cardio.eigen<-eigen(cardio.cov)
cardio.eigen$values
head(cardio.eigen$vectors)


xxx<-cardio.s # for easier references 

xxx.pca1<-prcomp(xxx, center=FALSE, scale.=FALSE) # stats::
xxx.pca1
xxx.pca1$rotation #only ‚Äúrotation‚Äù part, the matrix of variable loadings

summary(xxx.pca1) # first 2 PCs explain just a small fraction of variance ‚Äì poor‚Ä¶

# difference between prcomp()and princomp()
# stats::princomp() uses the spectral decomposition approach 
# stats::prcomp() and FactoMineR::PCA() use the singular value decomposition (SVD).

#xxx.pca2<-princomp(xxx) # stats::princomp()
#loadings(xxx.pca2)

library(factoextra)

plot(xxx.pca1)# the same will be plot(xxx.pca2)
fviz_pca_var(xxx.pca1, col.var="steelblue")# Corr plot, for xxx.pca2 looks similar

# It shows the relationships between all variables. It can be interpreted as follow:

# Positively correlated variables are grouped together.
# Negatively correlated variables are positioned on opposite sides of the plot origin (opposed quadrants).

# The distance between variables and the origin measures the quality of the variables on the factor map. Variables that are away from the origin are well represented on the factor map.

# visusalisation of quality
fviz_eig(xxx.pca1, choice='eigenvalue') # eigenvalues on y-axis
fviz_eig(xxx.pca1) # percentage of explained variance on y-axis

eig.val<-get_eigenvalue(xxx.pca1)
eig.val

a<-summary(xxx.pca1)
plot(a$importance[3,],type="l") # cumulative variance

# displaying the most significant variables that constitute PC ‚Äì here for PC1
loading_scores_PC_1<-xxx.pca1$rotation[,1]
fac_scores_PC_1<-abs(loading_scores_PC_1)
fac_scores_PC_1_ranked<-names(sort(fac_scores_PC_1, decreasing=T))
xxx.pca1$rotation[fac_scores_PC_1_ranked, 1]

# individual results with factoextra::
ind<-get_pca_ind(xxx.pca1)  
print(ind)

# The components of the get_pca_var() can be used in the plot of variables as follow:
# var$coord: coordinates of variables to create a scatter plot
# var$cos2: represents the quality of representation for variables on the factor map. It‚Äôs calculated as the squared coordinates: var.cos2 = var.coord * var.coord.
# var$contrib: contains the contributions (in percentage) of the variables to the principal components. The contribution of a variable (var) to a given principal component is (in percentage) : (var.cos2 * 100) / (total cos2 of the component).

head(ind$coord) # coordinates of variables

head(ind$contrib) # contributions of individuals to PC

# contributions of individual variables to PC
library(gridExtra)
var<-get_pca_var(xxx.pca1)
a<-fviz_contrib(xxx.pca1, "var", axes=1, xtickslab.rt=90) # default angle=45¬∞
b<-fviz_contrib(xxx.pca1, "var", axes=2, xtickslab.rt=90)
grid.arrange(a,b,top='Contribution to the first two Principal Components')

install.packages("rgl")
#WYSZLO!!!!!, TU SKONCZYLEM
card<-read.csv2("oldml.csv", sep=",")
library(pca3d)
card <- na.omit(card)
pca3d(xxx.pca1, group=as.factor(card$TenYearCHD))

########################################

# rotated PCA

# Using the psych :: package and the principal() command, you can rotate factors (loadings). 
# The aim of the rotation is to change the structure of the loadings to a simpler one (and easier in interpretation). 
# The rotation is around the axis of the factors - PCA maximizes the variance along the axis. 
# The principal() command has several rotation options. The most-used ones are: 
# - varimax - simplifies the interpretation of factors by minimizing the number of variables necessary to explain a given factor 
# - quartimax - contrary to varimax, simplifies the interpretation of variables by minimizing the factors needed to explain a given variable

# In summary(), one tests the hypothesis about the sufficient number of factors specified in the principal()
# By displaying loadings in a limited way (with cut-off) and sorted, you can interpret the influence of variables on factors.

# Finally, you can build new synthetic variables from the variables that are important under the factor (eg expressed as an arithmetic mean) and use them in further analyzes.

# PCA is essentially a rotation of the coordinate axes, chosen such that each successful axis captures as much variance as possible.
# In some disciplines (such as e.g. psychology), people like to apply PCA in order to interpret the resulting axes. I.e. they want to be able to say that principal axis #1 (which is a certain linear combination of original variables) has some particular meaning. To guess this meaning they would look at the weights in the linear combination. However, these weights are often messy and no clear meaning can be discerned.
# In these cases, people sometimes choose to tinker a bit with the vanilla PCA solution. They take certain number of principal axes (that are deemed "significant" by some criterion), and additionally rotate them, trying to achieve some "simple structure" --- that is, linear combinations that would be easier to interpret. There are specific algorithms that look for the simplest possible structure; one of them is called varimax. After varimax rotation, successive components do not anymore capture as much variance as possible! This feature of PCA gets broken by doing the additional varimax (or any other) rotation.
# So before applying varimax rotation, you have "unrotated" principal components. And afterwards, you get "rotated" principal components. In other words, this terminology refers to the post-processing of the PCA results and not to the PCA rotation itself.

#install.packages("psych")
library(psych)
xxx.pca4<-principal(xxx, nfactors=3, rotate="varimax")
xxx.pca4

summary(xxx.pca4)

# printing only the significant loadings
print(loadings(xxx.pca4), digits=3, cutoff=0.4, sort=TRUE)

# We should look for groups of RC and try to give ‚Äúumbrella name‚Äù for all components
# In this example it is hard ‚Äì maybe RC2 goes easiest as mainly collection of textile and services (except some meat)
# What important, many products are not important at all in explaining the changes ‚Äì see rows without numbers
# Anyway, those data are difficult to reduce (see low cumulative variance at RC3) ‚Äì no wonder, they are selected by Central Statistical Office to monitor inflation rate and they were selected well!

# This is the output which is the most interesting one for most of analysts

# Quality measures
# SCREE PLOTS
# we can check the explained variance with consecutive principal components (PC)

plot(xxx.pca1)
plot(xxx.pca1, type = "l")
fviz_eig(xxx.pca1) # factoextra::  

# Complexity - the higher this value, the more factor loads take values greater than zero. If the loading of only one factor is relatively large and the remaining ones are close to zero, the complexity is close to 1. In other words, how many variables constitute single factor. High complexity is an undesirable feature because it involves a more difficult interpretation of factors.

xxx.pca4$complexity

#install.packages("maptools")
library(maptools)
plot(xxx.pca4$complexity)

plot(xxx.pca4$complexity, pch=".", xlim=c(-20, 110), main="Complexity of factors ‚Äì keep it low", sub="How many variables constitute single factor.
The higher the number, the higher the (undesired) complexity", xlab=" ", ylab="complexity")
pointLabel(xxx.pca4$complexity, labels=names(xxx.pca4$complexity), cex=0.8) # maptools:

# Uniquenesses is the proportion of variance that is not shared with other variables. In PCA we want it be low, because then it is easier to reduce the space to a smaller number of dimensions. This means that the variable does not carry additional information in relation to other variables in the model. 

xxx.pca4$uniqueness

plot(xxx.pca4$uniqueness)

plot(xxx.pca4$uniqueness, pch=".", xlim=c(-20, 110), main="Uniqueness of factors ‚Äì keep it low", sub="Proportion of variance that is not shared with other variables.
The higher the number, the higher the (undesired) uniquenss", xlab=" ", ylab="complexity")
pointLabel(xxx.pca4$uniqueness, labels=names(xxx.pca4$uniqueness), cex=0.8) # maptools:

# linking uniquess and complexity

plot(xxx.pca4$complexity, xxx.pca4$uniqueness)

plot(xxx.pca4$complexity, xxx.pca4$uniqueness, xlim=c(0, 4))
pointLabel(xxx.pca4$complexity, xxx.pca4$uniqueness, labels=names(xxx.pca4$uniqueness), cex=0.8) # maptools:
abline(h=c(0.38, 0.75), lty=3, col=2)
abline(v=c(1.8), lty=3, col=2)

set<-data.frame(complex=xxx.pca4$complexity, unique=xxx.pca4$uniqueness)
set.worst<-set[set$complex>1.8 & set$unique>0.78,]
set.worst

# ‚ÄúWorst variables‚Äù are problematic in analysis, so if not necessary, better remove them

# visualization

# labeled observations in two dimensions
fviz_pca_ind(xxx.pca1, col.ind="#00AFBB", repel=TRUE)

# unlabeled observations in two dimensions with coloured quality of representation
fviz_pca_ind(xxx.pca1, col.ind="cos2", geom="point", gradient.cols=c("white", "#2E9FDF", "#FC4E07" ))

# colour correlation plot
fviz_pca_var(xxx.pca1, col.var = "steelblue")

# Another function based on ggplot2 ‚Äì autoplot()
#install.packages("ggfortify")
library(ggfortify)
autoplot(xxx.pca1)

autoplot(xxx.pca1, loadings=TRUE, loadings.colour='blue', loadings.label=TRUE, loadings.label.size=3)

# automatic colours by groups from other variable
# see similarity with clustering output
library(factoextra)
km3<-eclust(cardio, k=10) # 10 clusters for observations
autoplot(xxx.pca1, data=km3, colour="cluster")

# see the difference
# first figure much better
# PCA for data (products in columns) make products as loadings
# dots in autoplot are 2d observations (reduced from many more)
autoplot(xxx.pca1, loadings=TRUE, loadings.colour='blue', loadings.label=TRUE, loadings.label.size=3)

# right figure ‚Äì much worse
# PCA for transposed data (products in rows) make observations as loadings
# dots in autoplot are 2d products (reduced from many more)
autoplot(prcomp(t(xxx), center=FALSE, scale.=FALSE), loadings=TRUE, loadings.colour='blue', loadings.label=TRUE, loadings.label.size=3)

# to check values of ‚Äúrotation matrix‚Äù and PCA values
autoplot(xxx.pca1$rotation)
autoplot(xxx.pca1$x)

# Nice colour pallettes 
#install.packages("wesanderson")
library(wesanderson)
wes_palette("GrandBudapest1")

# interactive plots
# however, with three principal components, 3d is still flat

#install.packages("pca3d")
library(pca3d)
xxx.pca1<-prcomp(xxx, center=TRUE, scale.=TRUE) # stats::
km3<-eclust(cardio, k=10) # 10 clusters for observations, from factoextra::
gr<-factor(km3$cluster) 
summary(gr) 
pca3d(xxx.pca1, group=gr) # to make an interactive plot
#pca2d(xxx.pca1, group=gr, legend="topleft")
#pca2d(xxx.pca1, group=gr, biplot=TRUE, biplot.vars=3) # with directions

# one can run the k-means for PCA result with ClusterR:: 
# in this example:
# ClusterR::KMeans_rcpp() runs standard k-means, but allows for multiple 
# initializations and specifying the centroid
# ClusterR::plot_2d()makes ggplot-like graphics for clusters
# one specifies: a) data, b) clusters, and c) centroids/medoids

#install.packages("ClusterR")
library(ClusterR)

# k-means for PCA result
xxx.cs<-center_scale(xxx) # use ClusterR:: package for center_scale()
xxx.pca<-princomp(xxx.cs)$scores[, 1:2] # stats:: package
km<-KMeans_rcpp(xxx.pca, clusters=2, num_init=5, max_iters = 100) # from ClusterR::
plot_2d(xxx.pca, km$clusters, km$centroids)

# one can compare result with running k-means on MDS result
xxx.cs<-center_scale(xxx) # use ClusterR:: package for center_scale()
dist.reg<-dist(xxx.cs) # as a main input we need distance between units
mds1<-cmdscale(dist.reg, k = 2)
km<-KMeans_rcpp(mds1, clusters=2, num_init=5, max_iters = 100) # from ClusterR::
plot_2d(mds1, km$clusters, km$centroids)

# predictions

# whatever we do in selecting PC (in standard approach), prediction uses all PC‚Ä¶
# in options one can use rank.=2 for 2PC ‚Äì however, it does not change the result, it only limits the printout

# new data
xxx.new<-matrix(runif(66*3, 0,1), nrow=3, ncol=66)
colnames(xxx.new)<-colnames(xxx)

# PCA with prcomp:: (non-rotated) & preditions
xxx.pca1<-prcomp(xxx, center=FALSE, scale.=FALSE) # stats::

library(FactoMineR)
pred<-predict(xxx.pca1, as.data.frame(xxx.new))

plot(xxx.pca1$rotation[,1], xxx.pca1$rotation[,2], xlim=c(-2, 2), ylim=c(-2, 2), pch=21, bg="blue")
points(pred[,1], pred[,2], cex=2, pch=21, bg="yellow")

# in rotated PCA one can select number of factors for model ‚Äì and prediction
# in rotated PCA setting nfactors matters for the result

# PCA with principal:: (rotated) & preditions
library(psych)
xxx.pca4<-principal(xxx, nfactors=2, rotate="varimax")
pred<-predict.psych(xxx.pca4, as.data.frame(xxx.new), xxx)

plot(xxx.pca4$loadings[,1], xxx.pca4$loadings[,2], pch=21, bg="blue")
points(pred[,1], pred[,2], cex=2, pch=21, bg="yellow")

# short summary
# 1.One can estimate PCA (normal or rotated) to see how many principal components (artificial variables) explain the same information and which original variables should be included in designing the model
# 2.Having 2D PCA, one can visualise the mutual relations between observations on the plot ‚Äì one can also cluster the output
# 3.Prediction is to generate new PCA-coordinates for original-format data. Standard procedure uses all principal components and all loadings
# 4.One can also run PCA-regression which goes step further, a limits dataset to put it into regression. 
