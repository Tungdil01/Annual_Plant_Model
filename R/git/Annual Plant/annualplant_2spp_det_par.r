library(snow)
cores=8
print(date())
# parameters -----------------------------------------
  # name of output file:
  outfile =  "annplant_2spp_det1.csv"

  #specify combinations of freq-dep parameters
  l1_v=10:20
  l2_v=10:20

  a11_v=c(0.1,0.3,0.5)
  a12_v=c(0.1,0.3,0.5,0.7,0.9,1)
  a21_v=c(0.1,0.3,0.5,0.7,0.9,1)
  a22_v=c(0.1,0.3,0.5,0.7,0.9,1)

  simul = expand.grid("l1"=l1_v,"l2"=l2_v,"a11"=a11_v,"a12"=a12_v,"a21"=a21_v,"a22"=a22_v)

#simulation output
simul$N1 = NA
simul$N2 = NA

cluster1 <- makeCluster(cores, type = "SOCK")
clusterCall(cluster1, function() source("analyN_function.r"))

#simulation function-------------------------------------
Sim = function(k, simul) {

    l1=simul$l1[k]           #get parameter combo
    l2=simul$l2[k]
    a11=simul$a11[k]
    a12=simul$a12[k]
    a21=simul$a21[k]
    a22=simul$a22[k]
    
#simulation stats
simul$N1[k] = analyN(l1,l2,a11,a12,a21,a22)[1]
simul$N2[k] = analyN(l1,l2,a11,a12,a21,a22)[2]

simul[k,]
}

k=seq(1:dim(simul)[1])
simul1=parSapply(cluster1, k, Sim, simul=simul)
simul1=t(simul1)
write.table(simul1,outfile, sep=",",row.names=FALSE)
stopCluster(cluster1)


#pdf(file="saving_plot.pdf")
# if only one run performed, make figures
if(dim(simul)[1]==1){
  par(mfrow=c(1,2),tcl=-0.2)
  # plot density time series
  matplot(N,type="l",xlab="Time",ylab="N",col=c("steelblue","firebrick"))
  # plot frequency dependence in growth
  Nfreq = N/rowSums(N)  # calculate frequencies
  Nfreq[Nfreq==1]=NA # remove values after one species goes exinct
  growth = log(N[2:NROW(N),])-log(N[1:(NROW(N)-1),])
  growth[growth==-Inf]=NA
  myLims = c(min(growth,na.rm=T)-0.05,max(growth,na.rm=T)+0.05)
  plot(Nfreq[1:NROW(Nfreq)-1,1],growth[,1],xlab="Frequency",ylab="Growth rate",
    xlim=c(0,1),ylim=myLims,col="steelblue")
  abline(lm(growth[,1]~ Nfreq[1:NROW(Nfreq)-1,1] ),col="steelblue")
  par(new=T)
  plot(Nfreq[1:NROW(Nfreq)-1,2],growth[,2],xlab="",ylab="",xaxt="n",yaxt="n",
    xlim=c(0,1),ylim=myLims,col="firebrick")
  abline(lm(growth[,2]~ Nfreq[1:NROW(Nfreq)-1,2] ),col="firebrick")
  abline(0,0,lty="dotted")
  legend("topright",legend=c("Spp 1", "Spp 2"),lty="solid",col=c("steelblue","firebrick"),bty="n")
}
#dev.off()
