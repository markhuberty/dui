import collections
import dedupe
import logging
import optparse
import os
import pandas as pd
import random
import sys
import time

optp = optparse.OptionParser()
optp.add_option('-v', '--verbose', dest='verbose', action='count',
                help='Increase verbosity (specify multiple times for more)'
                )
(opts, args) = optp.parse_args()
log_level = logging.WARNING 
if opts.verbose == 1:
    log_level = logging.INFO
elif opts.verbose >= 2:
    log_level = logging.DEBUG
logging.basicConfig(level=log_level)

inputs = [i for idx, i in enumerate(sys.argv) if idx > 0]
input_file = inputs[0]


## Set the constants for blocking
ppc=0.001
dupes=5
recall_wt = 1.5
settings_file = 'alc_dedupe_settings.json'
training_file = 'alc_dedupe_training.json'
output_file = '../data/alc_dedupe_output.csv'

def readDataFrame(df):
    data_d = {}

    for idx, dfrow in df.iterrows():
        row_out = {}

        name_first = dfrow['subfirstname'].lower()
        name_last = dfrow['sublastname'].lower()
        name_mi = dfrow['submi'].lower()

        row_out['first'] = name_first
        row_out['last'] =  name_last
        row_out['mi'] = name_mi

        row_tuple = [(k, v) for (k, v) in row_out.items()]
        data_d[idx] = dedupe.core.frozendict(row_tuple)

    return data_d


def return_threshold_data(blocks, n_samples=1000):
    """
    Given a block map and a corresponding data object, return
    n_samples random blocks as a list of tuples of form
    (record_id, record)
    """
    blocks_sub = [b for b in blocks if len(b) > 1]

    if n_samples > len(blocks_sub):
        n_samples = len(blocks_sub)
    subset = random.sample(range(len(blocks_sub)), n_samples)

    threshold_data = []
    for idx in subset:
        threshold_data.append(blocks_sub[idx])
    return tuple(threshold_data)


# Read in the data
input_df = pd.read_csv(input_file)
data_d = readDataFrame(input_df)


# Training
time_start = time.time()
if os.path.exists(settings_file):
    print 'reading from', settings_file
    deduper = dedupe.Dedupe(settings_file)

else:
    # To train dedupe, we feed it a random sample of records.
    data_sample = dedupe.dataSample(data_d, 10 * input_df.shape[0])
    # Define the fields dedupe will pay attention to

    fields = {'first': {'type': 'String'},
              'last': {'type': 'String'},
              'mi':{'type': 'String'}
              }

    # Create a new deduper object and pass our data model to it.
    deduper = dedupe.Dedupe(fields)

    if os.path.exists(training_file):
        print 'reading labeled examples from ', training_file
        deduper.train(data_sample, training_file)

    # ## Active learning

    # Starts the training loop. Dedupe will find the next pair of records
    # it is least certain about and ask you to label them as duplicates
    # or not.

    # use 'y', 'n' and 'u' keys to flag duplicates
    # press 'f' when you are finished
    print 'starting active labeling...'
    deduper.train(data_sample, dedupe.training.consoleLabel)

    # When finished, save our training away to disk
    deduper.writeTraining(training_file)

print 'blocking...'
blocking_time_start = time.time()
blocker = deduper.blockingFunction(ppc, dupes)

# Occassionally the blocker fails to find useful values. If so,
# print the final values and exit.
if not blocker:
    print 'No valid blocking settings found'
    print 'Starting ppc value: %s' % ppc
    print 'Starting uncovered_dupes value: %s' % dupes
    print 'Exiting'
    sys.exit()

time_block_weights = time.time()
print 'Learned blocking weights in', time_block_weights - blocking_time_start, 'seconds'

# Save weights and predicates to disk.
# If the settings file exists, we will skip all the training and learning
deduper.writeSettings(settings_file)

# Generate the tfidf canopy
## NOTE: new version of blockData does tfidf implicitly
# print 'generating tfidf index'
# full_data = ((k, data_d[k]) for k in data_d)
# blocker.tfIdfBlocks(full_data)
# del full_data

# Load all the original data in to memory and place
# them in to blocks. Return only the block_id: unique_id keys
#blocking_map = patent_util.return_block_map(data_d, blocker)

# Note this is now just a tuple of blocks, each of which is a
# recordid: record dict

blocked_data = dedupe.blockData(data_d, blocker)
#keys_to_block = [k for k in blocking_map if len(blocking_map[k]) > 1]
print '# Blocks to be clustered: %s' % len(blocked_data)

# Save the weights and predicates
time_block = time.time()
print 'Blocking rules learned in', time_block - time_block_weights, 'seconds'
print 'Writing out settings'
deduper.writeSettings(settings_file)

# ## Clustering

# Find the threshold that will maximize a weighted average of our precision and recall. 
# When we set the recall weight to 1, we are trying to balance recall and precision
#
# If we had more data, we would not pass in all the blocked data into
# this function but a representative sample.

threshold_data = return_threshold_data(blocked_data, 10000)

print 'Computing threshold'
threshold = deduper.goodThreshold(threshold_data, recall_weight=recall_wt)
del threshold_data

# `duplicateClusters` will return sets of record IDs that dedupe
# believes are all referring to the same entity.



print 'clustering...'
# Loop over each block separately and dedupe

clustered_dupes = deduper.duplicateClusters(blocked_data,
                                            threshold
                                            ) 

print '# duplicate sets', len(clustered_dupes)

# Extract the new cluster membership
max_cluster_id = 0
cluster_membership = collections.defaultdict(lambda : 'x')
for (cluster_id, cluster) in enumerate(clustered_dupes):
    for record_id in cluster:
        cluster_membership[record_id] = cluster_id
        if max_cluster_id <= cluster_id:
            max_cluster_id  = cluster_id + 1

# Then write it into the data frame as a sequential index for later use
# Here the cluster ID is either the dedupe ID (if a PATSTAT person belonged
# to a block of 2 or more potential matches) or an integer ID placeholder.

cluster_index = []
clustered_cluster_map = {}
excluded_cluster_map = {}
for df_idx in input_df.index:
    if df_idx in cluster_membership:
        orig_cluster = cluster_membership[df_idx]
        if orig_cluster in clustered_cluster_map:
            cluster_index.append(clustered_cluster_map[orig_cluster])
        else:
            clustered_cluster_map[orig_cluster] = max_cluster_id
            cluster_index.append(max_cluster_id) #cluster_counter)
            max_cluster_id += 1
            # print cluster_counter
    else:
        if df_idx in excluded_cluster_map:
            cluster_index.append(excluded_cluster_map[df_idx])
        else:
            excluded_cluster_map[df_idx] = max_cluster_id
            cluster_index.append(max_cluster_id)
            max_cluster_id += 1

cluster_name = 'cluster_id'
input_df[cluster_name] = cluster_index

# Write out the data frame
input_df.to_csv(output_file)
print 'Dedupe complete, ran in ', time.time() - time_start, 'seconds'
