#!/usr/bin/env Rscript
# Population orientation tracks with DIRECTIONAL COLOR GRADIENT.
# Each haplotype row: alignment binned; each bin colored by its QUERY coordinate
# (viridis). Forward alignment -> gradient ascends L->R; inverted -> gradient
# reverses. Inversions read instantly as a flipped color ramp.
suppressPackageStartupMessages({library(SVbyEye);library(ggplot2);library(GenomicRanges)})
a<-commandArgs(trailingOnly=TRUE)
paf_file<-a[1];out<-a[2];inv_id<-a[3];chrom<-a[4]
istart<-as.integer(a[5]);iend<-as.integer(a[6]);wstart<-as.integer(a[7]);wend<-as.integer(a[8]);af<-a[9]
VIR<-c("#440154","#414487","#2a788e","#22a884","#7ad151","#fde725")
pt<-readPaf(paf_file,include.paf.tags=TRUE,restrict.paf.tags="cg")
pt<-pt[pt$t.name==chrom,]
pt<-tryCatch(subsetPafAlignments(pt,target.region=GRanges(chrom,IRanges(wstart,wend))),error=function(e)pt)
pt<-tryCatch(filterPaf(pt,min.align.len=2000),error=function(e)pt)
if(nrow(pt)==0){ggsave(out,ggplot()+annotate("text",x=.5,y=.5,label=paste(inv_id,"empty"))+theme_void(),width=10,height=3);quit()}
# classify + order
L<-iend-istart; clip<-function(a0,a1,b0,b1){pmax(0,pmin(a1,b1)-pmax(a0,b0))}
qn<-unique(pt$q.name); fr<-sapply(qn,function(q){s<-pt[pt$q.name==q,];ov<-clip(s$t.start,s$t.end,istart,iend)
  co<-sum(ov); if(co==0) return(-1); sum(ov[s$strand=="-"])/co})
isch<-grepl("panTro|chimp",qn,ignore.case=TRUE)
ok<-sapply(qn,function(q){s<-pt[pt$q.name==q,];sum(clip(s$t.start,s$t.end,istart,iend))>=0.3*L})
qn<-qn[ok];fr<-fr[ok];isch<-isch[ok]
ord<-order(ifelse(isch,-1,fr)); lev<-qn[ord]
# bin each record; compute query coord per bin
NB<-160; binw<-(wend-wstart)/NB
rows<-list()
for(i in seq_along(pt$q.name)){
  q<-pt$q.name[i]; if(!q %in% lev) next
  ts<-pt$t.start[i];te<-pt$t.end[i];qs<-pt$q.start[i];qe<-pt$q.end[i];st<-pt$strand[i]
  ts2<-max(ts,wstart);te2<-min(te,wend); if(te2<=ts2) next
  b0<-floor((ts2-wstart)/binw); b1<-ceiling((te2-wstart)/binw)
  for(b in b0:(b1-1)){
    x0<-wstart+b*binw; x1<-x0+binw; xm<-(max(x0,ts)+min(x1,te))/2
    frac<-(xm-ts)/(te-ts); qpos<-if(st=="-") qe-frac*(qe-qs) else qs+frac*(qe-qs)
    rows[[length(rows)+1]]<-data.frame(q=q,x0=max(x0,wstart),x1=min(x1,wend),qpos=qpos,y=match(q,lev))
  }
}
df<-do.call(rbind,rows)
# normalize qpos per haplotype to 0..1 (query is contig-native; per-hap ramp)
# align each haplotype's query origin to the reference window, keep reference scale
span<-(wend-wstart)
df$qn01<-ave(seq_len(nrow(df)),df$q,FUN=function(ix){v<-df$qpos[ix];mn<-min(v);pmax(0,pmin(1,(v-mn)/span))})
showlab<-length(lev)<=45
lab<-ifelse(grepl("panTro|chimp",lev,ignore.case=TRUE),paste0("* ",lev),lev)
kb<-round(L/1000,1)
p<-ggplot(df)+
  annotate("rect",xmin=istart,xmax=iend,ymin=.3,ymax=length(lev)+.7,fill=NA,color="grey35",linetype="dashed",linewidth=.3)+
  geom_rect(aes(xmin=x0,xmax=x1,ymin=y-.42,ymax=y+.42,fill=qn01))+
  scale_fill_gradientn(colours=VIR,name="query position\n(gradient = alignment direction)")+
  scale_y_continuous(breaks=if(showlab)seq_along(lev)else waiver(),labels=if(showlab)lab else waiver(),expand=expansion(add=.6))+
  scale_x_continuous(labels=function(x)paste0(round(x/1e6,2),"Mb"))+
  labs(title=sprintf("%s  %s:%s-%s  %skb  AF=%s  (%d haplotypes; gradient flips at inversions)",inv_id,chrom,format(istart,big.mark=","),format(iend,big.mark=","),kb,af,length(lev)),x=chrom,y=NULL)+
  theme_bw()+theme(plot.title=element_text(size=8),axis.text.y=element_text(size=5),legend.position="bottom",panel.grid.minor=element_blank())
ggsave(out,p,width=11,height=max(3,min(30,length(lev)*.14+2)),dpi=150,limitsize=FALSE)
cat("WROTE",out,"\n")
