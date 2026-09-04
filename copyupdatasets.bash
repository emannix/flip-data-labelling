

# SIZE=$(du -sb original_master_2026_09_04 | awk '{print $1}')
# time tar -cf - original_master_2026_09_04 | pv -s "$SIZE" | pigz -0 > original_master_2026_09_04.tar.gz
# # copy to mediaflux
# rsync -ahP --info=progress2 original_master_2026_09_04.tar.gz mnt/farm_classification/data/original_new_2026/
# rsync -ahP --info=progress2 original_master_2026_09_04.tar.gz emannix@spartan.hpc.unimelb.edu.au:/data/cephfs/punim1019/FLIP-phase-3/data/

