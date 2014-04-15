
#model<- ergm(g~triangle+nodematch(attrname="R",diff=T,keep=c(2)),constraints=~edges)

library("statnet")
library(ergm)
library(sna)


in_<-"C:\\Users\\wu\\Desktop\\15-15_threshold_0.3\\"
out_<-"C:\\Users\\wu\\Desktop\\ERGM_coefficients.txt"

write("Healthy",file=out_,append=T)

healthy_filenames <- list.files(paste(in_,"Healthy",sep=""),full.name=TRUE)
readPaj<- function(x) read.paj(x)
healthy_graphlist<-lapply(healthy_filenames,readPaj)

for(i in 1:length(healthy_graphlist)){

g<-healthy_graphlist[[i]]
set.vertex.attribute(g,attrname="R",value=c(0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1))
print ("fitting model")
print (i)

model<- ergm(g~kstar(2)+triangle+nodematch(attrname="R",diff=F,keep=c(2)),constraints=~edges)
density<-network.density(g)
vars<-as.numeric(c(density,coef(model)))
write(vars,file=out_,append=T)
print(summary(model))

}


write("Disease",file=out_,append=T)


disease_filenames <- list.files(paste(in_,"Disease",sep=""),full.name=TRUE)
readPaj<- function(x) read.paj(x)
disease_graphlist<-lapply(disease_filenames,readPaj)



for(i in 1:length(disease_graphlist)){

g<-disease_graphlist[[i]]
set.vertex.attribute(g,attrname="R",value=c(0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1))
print ("fitting model")
print (i)

model<- ergm(g~kstar(2)+triangle+nodematch(attrname="R",diff=F,keep=c(2)),constraints=~edges)
density<-network.density(g)
vars<-as.numeric(c(density,coef(model)))
write(vars,file=out_,append=T)
print(summary(model))
}

