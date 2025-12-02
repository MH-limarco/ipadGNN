for methood in "gnn" "prune_gnn" ; do
  for model in "GCN" "SAGE" ; do
      sh miss_gcn.sh $methood $model
      sh miss_gcn_classic.sh $methood $model
  done
done