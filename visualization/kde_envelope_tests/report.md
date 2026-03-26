# KDE Envelope Test Report

## Case 1: clean A/B clusters

- Plot: `/storage/BioMedNLP/llm2vec/visualization/kde_envelope_tests/case1_clean_two_clusters.png`
- Summary: `/storage/BioMedNLP/llm2vec/visualization/kde_envelope_tests/case1_clean_two_clusters_summary.csv`

label,points,eps,dbscan_clusters,drawn_kde_clusters,ignored_small_clusters,noise_points
A,180,0.1208,1,1,0,14
B,170,0.1357,1,1,0,12

## Case 2: A outliers inside B cluster

- Plot: `/storage/BioMedNLP/llm2vec/visualization/kde_envelope_tests/case2_a_outliers_inside_b.png`
- Summary: `/storage/BioMedNLP/llm2vec/visualization/kde_envelope_tests/case2_a_outliers_inside_b_summary.csv`

label,points,eps,dbscan_clusters,drawn_kde_clusters,ignored_small_clusters,noise_points
A,186,0.1794,2,1,1,6
B,175,0.1517,1,1,0,15

## Case 3: A has two major islands and one tiny island

- Plot: `/storage/BioMedNLP/llm2vec/visualization/kde_envelope_tests/case3_a_two_major_islands.png`
- Summary: `/storage/BioMedNLP/llm2vec/visualization/kde_envelope_tests/case3_a_two_major_islands_summary.csv`

label,points,eps,dbscan_clusters,drawn_kde_clusters,ignored_small_clusters,noise_points
A,192,0.2383,3,2,1,2
B,155,0.1206,1,1,0,18
