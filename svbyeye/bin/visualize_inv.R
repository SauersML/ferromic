#!/usr/bin/env Rscript
# Per-inversion visualization, two complementary views:
#  (1) <prefix>.png       SVbyEye plotMiro miropeat on a small curated set
#                         (chimp + K direct + K inverted) -> clean canonical view
#  (2) <prefix>.track.png stacked orientation tracks for ALL assessed haplotypes
#                         (one row per haplotype, colored by strand vs hg38)
suppressPackageStartupMessages({library(SVbyEye);library(ggplot2);library(GenomicRanges)})
a<-commandArgs(trailingOnly=TRUE)
paf_file<-a[1]; out_prefix<-a[2]; inv_id<-a[3]; chrom<-a[4]
istart<-as.integer(a[5]); iend<-as.integer(a[6]); wstart<-as.integer(a[7]); wend<-as.integer(a[8])
af<-a[9]; recur<-a[10]; disease<-a[11]
K<-2  # miropeat: max direct & max inverted haplotypes (keep it clean)
FILL<-c("+"="#2e8b3d","-"="#4f7fd4")

placeholder<-function(msg){
  p<-ggplot()+annotate("text",x=.5,y=.5,label=paste0(inv_id,"\n",msg),size=6)+theme_void()
  ggsave(paste0(out_prefix,".png"),p,width=10,height=4,dpi=150)
  cat("PLACEHOLDER",inv_id,msg,"\n"); quit(save="no",status=0)
}
if(!file.exists(paf_file)||file.info(paf_file)$size==0) placeholder("no overlapping alignments")
pt<-tryCatch(readPaf(paf_file,include.paf.tags=TRUE,restrict.paf.tags="cg"),
             error=function(e)data.frame())
if(nrow(pt)==0) placeholder("empty PAF")
pt<-pt[pt$t.name==chrom,]; if(nrow(pt)==0) placeholder("no records on target chrom")
gr_win<-GRanges(chrom,IRanges(wstart,wend)); gr_inv<-GRanges(chrom,IRanges(istart,iend))
pt<-tryCatch(subsetPafAlignments(pt,target.region=gr_win),error=function(e)pt)
pt<-tryCatch(filterPaf(pt,min.align.len=2000),error=function(e)pt)
if(nrow(pt)==0) placeholder("nothing in window after filter")

# classify each haplotype by reverse-coverage across the inversion interval
L<-iend-istart
clip<-function(a0,a1,b0,b1){lo<-pmax(a0,b0);hi<-pmin(a1,b1);pmax(0,hi-lo)}
qn<-unique(pt$q.name)
cls<-data.frame(q=qn,cov=0,rev=0,stringsAsFactors=FALSE)
for(i in seq_along(qn)){s<-pt[pt$q.name==qn[i],];ov<-clip(s$t.start,s$t.end,istart,iend)
  cls$cov[i]<-sum(ov);cls$rev[i]<-sum(ov[s$strand=="-"])}
cls$frac_rev<-ifelse(cls$cov>0,cls$rev/cls$cov,0)
cls$assessed<-cls$cov>=0.30*L
cls$is_chimp<-grepl("panTro|chimp",cls$q,ignore.case=TRUE)
cls$class<-ifelse(!cls$assessed,"low",ifelse(cls$is_chimp,"chimp",
            ifelse(cls$frac_rev>0.6,"inverted",ifelse(cls$frac_rev<0.4,"direct","ambig"))))
n_dir<-sum(cls$class=="direct");n_inv<-sum(cls$class=="inverted");n_amb<-sum(cls$class=="ambig")
n_hap<-sum(cls$assessed & !cls$is_chimp); obs<-if(n_hap>0) n_inv/n_hap else NA
obs_s<-if(is.na(obs))"NA" else sprintf("%.2f",obs)
kb<-round(L/1000,1)
ttl<-sprintf("%s  %s:%s-%s  %skb  AF_exp=%s obs=%s [dir=%d inv=%d amb=%d n=%d]",
  inv_id,chrom,format(istart,big.mark=","),format(iend,big.mark=","),kb,af,obs_s,n_dir,n_inv,n_amb,n_hap)
subt<-if(disease!="NA"&&nzchar(disease)) paste("genes:",disease) else NULL

## ---- (1) SVbyEye miropeat on small curated set ----
pick<-function(df,n) df$q[order(-df$cov)][seq_len(min(n,nrow(df)))]
sel<-c(cls$q[cls$class=="chimp"],pick(cls[cls$class=="direct",,drop=F],K),
       pick(cls[cls$class=="ambig",,drop=F],1),pick(cls[cls$class=="inverted",,drop=F],K))
if(length(sel)==0) sel<-pick(cls,6)
pm<-pt[pt$q.name %in% sel,]
ord<-rev(c(cls$q[cls$class=="chimp"],pick(cls[cls$class=="direct",,drop=F],K),
       pick(cls[cls$class=="ambig",,drop=F],1),pick(cls[cls$class=="inverted",,drop=F],K)))
pm$q.name<-factor(pm$q.name,levels=ord); pm<-pm[!is.na(pm$q.name),]; pm$q.name<-as.character(pm$q.name)
plt<-tryCatch(plotMiro(paf.table=pm,color.by="direction"),error=function(e)NULL)
if(!is.null(plt)){
  plt<-tryCatch(addAnnotation(plt,annot.gr=gr_inv,coordinate.space="target",shape="rectangle",annotation.label="INV"),error=function(e)plt)
  plt<-plt+labs(title=ttl,subtitle=subt)+theme(plot.title=element_text(size=8),plot.subtitle=element_text(size=7),legend.position="bottom")
  nn<-length(unique(pm$q.name)); ggsave(paste0(out_prefix,".png"),plt,width=12,height=max(3.5,min(14,nn*0.55+2.5)),dpi=150,limitsize=FALSE)
  tryCatch(ggsave(paste0(out_prefix,".pdf"),plt,width=12,height=max(3.5,min(14,nn*0.55+2.5)),limitsize=FALSE),error=function(e)NULL)
} else placeholder("plotMiro failed")

## ---- (2) stacked orientation tracks: ALL assessed haplotypes ----
keep<-cls$q[cls$assessed]
if(length(keep)>0){
  seg<-pt[pt$q.name %in% keep,c("q.name","t.start","t.end","strand")]
  # clip segments to window for display
  seg$t.start<-pmax(seg$t.start,wstart); seg$t.end<-pmin(seg$t.end,wend)
  seg<-seg[seg$t.end>seg$t.start,]
  # order: chimp top, then direct (low frac_rev) ... inverted (high frac_rev)
  o<-cls[cls$assessed,]; o$key<-ifelse(o$is_chimp,-1,o$frac_rev)
  o<-o[order(o$key),]
  lev<-o$q; seg$y<-match(seg$q.name,lev)
  lab<-ifelse(o$is_chimp,paste0("* ",o$q),o$q)
  showlab<-length(lev)<=45
  gt<-ggplot(seg)+
    annotate("rect",xmin=istart,xmax=iend,ymin=0.3,ymax=length(lev)+0.7,fill="grey85",alpha=.5)+
    geom_vline(xintercept=c(istart,iend),linetype="dashed",color="grey40",linewidth=.3)+
    geom_rect(aes(xmin=t.start,xmax=t.end,ymin=y-0.42,ymax=y+0.42,fill=strand))+
    scale_fill_manual(values=FILL,name="strand vs hg38",labels=c("+"="+ (same)","-"="- (inverted)"))+
    scale_y_continuous(breaks=if(showlab)seq_along(lev)else waiver(),labels=if(showlab)lab else waiver(),expand=expansion(add=.5))+
    scale_x_continuous(labels=function(x)paste0(round(x/1e6,2),"Mb"))+
    labs(title=ttl,subtitle=paste0(subt,if(!is.null(subt))"  |  " else "","* = chimp (ancestral);  ",length(lev)," haplotypes"),x=chrom,y=NULL)+
    theme_bw()+theme(plot.title=element_text(size=8),plot.subtitle=element_text(size=7),
      axis.text.y=element_text(size=5),legend.position="bottom",panel.grid.minor=element_blank())
  ht<-max(3,min(30,length(lev)*0.14+2))
  ggsave(paste0(out_prefix,".track.png"),gt,width=11,height=ht,dpi=150,limitsize=FALSE)
}
cat(sprintf("QC\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%s\t%s\n",inv_id,chrom,istart,iend,n_hap,n_dir,n_inv,obs_s,af))
cat("WROTE",out_prefix,"\n")
