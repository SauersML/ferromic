import sys, os
import msprime

#Tsp_p0_p1			 /\    
#				  D /  \ I
#				   /    \ 
#				  /	     \ 
#				D	       I
#			   p0			p1


# First we set out the fixed values of the various parameters
N_a = 6000
mu = 1.25e-8
generation_time = 25

# Times are provided in years, so we convert into generations.
Tsp_p0_p1 = int(sys.argv[1]) / generation_time

# sample sizes of haplotypes in each pop/deme
sample_pop0, sample_pop1 = eval(sys.argv[2])

# Recombination and migration rates
rho = float(sys.argv[3])
m_const = float(sys.argv[4])
seq_length = int(sys.argv[5])
chromID = sys.argv[6]

samples = [msprime.Sample(0,0) for i in range(sample_pop0)] + [msprime.Sample(1,0) for i in range(sample_pop1)] 

haploIDs = []
for i, v in enumerate(eval(sys.argv[2])):
	for ii in range(v):
		if i not in [1]:
			haploIDs.append("D%s%s" % (i, ii))
		else:
			haploIDs.append("I%s%s" % (i, ii))

sampleIDs = []
for i in range(0, len(haploIDs),2):
	sampleIDs.append("%s_%s" % (haploIDs[i], haploIDs[i+1]))



# Population IDs correspond to their indexes in the population
# configuration array. Therefore, we have 0=pop0, 1=pop1, 2=pop2, and 3=pop3
# initially.
pop_config=[ 
	msprime.PopulationConfiguration(sample_size=None, initial_size=N_a),
	msprime.PopulationConfiguration(sample_size=None, initial_size=N_a/100)]

mig_mat=[
	[0, 0],  
	[0, 0]]


demographic_events=[
	# Finally merge pop1 to pop3 at Tsp_p0_p1
	msprime.MassMigration(time=Tsp_p0_p1, source=1, destination=0, proportion=1.0)]

#dd = msprime.DemographyDebugger(Ne=1, population_configurations = pop_config, migration_matrix = mig_mat, demographic_events = demographic_events)
#dd.print_history()

treeseq = msprime.simulate(population_configurations = pop_config, samples = samples, migration_matrix = mig_mat, demographic_events = demographic_events, length = seq_length, recombination_rate = rho, mutation_rate = mu)


with sys.stdout as vcffile:
	treeseq.write_vcf(vcffile, contig_id = chromID, ploidy=2, individual_names=sampleIDs )


