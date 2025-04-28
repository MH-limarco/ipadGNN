
for methood in "gnn" "prune_gnn" ; do
  sh miss_gcn.sh $methood
  sh miss_gcn_classic.sh $methood
done