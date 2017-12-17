# MT: Code

Docker auf srv-lab-t-697 starten:
    
    scr 3 # bzw. irgendeine Zahl (erstekllt eine Screen-Session)
    srun --pty --ntasks=1 --cpus-per-task=1 --mem=32G --gres=gpu:1 bash
    # if required: docker build -t keras-meierbe8-4 images/keras4
    nvidia-docker run -it -v ~/data:/data -v ~/code/:/code -v ~/code/tmp:/src/tmp -v ~ keras-meierbe8-4 bash
    pip install termcolor librosa yattag
    cd tmp/ClusterNN
    ln -s /data/MT/ /tmp/test
    (cd /home/keras/; mv .keras .keras_org; ln -s /data/.keras .)
    bash ./scripts/run.sh ./app/test_minimal_cluster_nn.py /tmp/test_minimal_cluster_nn.py.log

    ./scripts/run.sh ./playground/trainable_depth.py /tmp/test/trainable_depth_000.log

    pip install termcolor librosa yattag
    cd tmp/ClusterNN
    ln -s /data/MT/ /tmp/test
    (cd /home/keras/; mv .keras .keras_org; ln -s /data/.keras .)
	cp /tmp/test/cudnn_recurrent.py /tmp/test/wrappers.py /opt/conda/lib/python3.5/site-packages/keras/layers/


Bzw. falls root & Port-Weiterleitung gewünscht ist:

    nvidia-docker run -it -p 8792:8888 -v ~/data:/data -v ~/code/:/code -v ~/code/tmp:/src/tmp keras-meierbe8-3 bash
    
# Commands
	
	bash ./scripts/run.sh ./app/test_minimal_cluster_nn.py /tmp/test/test_minimal_cluster_nn.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00.py ~/data/MT/test_cluster_nn_try00.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_noembd.py ~/data/MT/test_cluster_nn_try00_noembd.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_noembd_v02.py ~/data/MT/test_cluster_nn_try00_noembd_v02.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_fixedc.py ~/data/MT/test_cluster_nn_try00_fixedc.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try01.py ~/data/MT/test_cluster_nn_try01.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try02.py ~/data/MT/test_cluster_nn_try02.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try02_mod01.py ~/data/MT/test_cluster_nn_try02_mod01.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try03_kmeans.py ~/data/MT/test_cluster_nn_try03_kmeans.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try03_kmeans_v02.py ~/data/MT/test_cluster_nn_try03_kmeans_v02.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try03_kmeans_v03.py ~/data/MT/test_cluster_nn_try03_kmeans_v03.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try03_kmeans_v04.py ~/data/MT/test_cluster_nn_try03_kmeans_v04.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try03_kmeans_v04b.py ~/data/MT/test_cluster_nn_try03_kmeans_v04b.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try03_kmeans_v05.py ~/data/MT/test_cluster_nn_try03_kmeans_v05.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v02.py ~/data/MT/test_cluster_nn_try00_v02.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v03.py ~/data/MT/test_cluster_nn_try00_v03.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v04.py ~/data/MT/test_cluster_nn_try00_v04.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v05.py ~/data/MT/test_cluster_nn_try00_v05.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v06.py ~/data/MT/test_cluster_nn_try00_v06.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v07.py ~/data/MT/test_cluster_nn_try00_v07.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v08.py ~/data/MT/test_cluster_nn_try00_v08.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v09.py ~/data/MT/test_cluster_nn_try00_v09.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v10.py ~/data/MT/test_cluster_nn_try00_v10.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v11.py ~/data/MT/test_cluster_nn_try00_v11.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v12.py ~/data/MT/test_cluster_nn_try00_v12.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v13.py ~/data/MT/test_cluster_nn_try00_v13.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v14.py ~/data/MT/test_cluster_nn_try00_v14.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v15.py ~/data/MT/test_cluster_nn_try00_v15.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v16.py ~/data/MT/test_cluster_nn_try00_v16.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v17.py ~/data/MT/test_cluster_nn_try00_v17.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try04_ddbc.py ~/data/MT/test_cluster_nn_try04_ddbc.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v18.py ~/data/MT/test_cluster_nn_try00_v18.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try04_ddbc_v02.py ~/data/MT/test_cluster_nn_try04_ddbc_v02.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v19.py ~/data/MT/test_cluster_nn_try00_v19.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v20.py ~/data/MT/test_cluster_nn_try00_v20.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v21.py ~/data/MT/test_cluster_nn_try00_v21.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v22.py ~/data/MT/test_cluster_nn_try00_v22.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v23.py ~/data/MT/test_cluster_nn_try00_v23.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v24.py ~/data/MT/test_cluster_nn_try00_v24.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v25.py ~/data/MT/test_cluster_nn_try00_v25.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v26.py ~/data/MT/test_cluster_nn_try00_v26.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v27.py ~/data/MT/test_cluster_nn_try00_v27.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v28.py ~/data/MT/test_cluster_nn_try00_v28.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v29.py ~/data/MT/test_cluster_nn_try00_v29.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v30.py ~/data/MT/test_cluster_nn_try00_v30.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v31.py ~/data/MT/test_cluster_nn_try00_v31.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v32.py ~/data/MT/test_cluster_nn_try00_v32.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v33.py ~/data/MT/test_cluster_nn_try00_v33.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v34.py ~/data/MT/test_cluster_nn_try00_v34.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v35.py ~/data/MT/test_cluster_nn_try00_v35.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v36.py ~/data/MT/test_cluster_nn_try00_v36.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v37.py ~/data/MT/test_cluster_nn_try00_v37.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v38.py ~/data/MT/test_cluster_nn_try00_v38.py.log
	# From now a keras upgrade may be required (>=2.0.8): pip install https://github.com/fchollet/keras/archive/master.zip
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v39.py ~/data/MT/test_cluster_nn_try00_v39.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v40.py ~/data/MT/test_cluster_nn_try00_v40.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v41.py ~/data/MT/test_cluster_nn_try00_v41.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v42.py ~/data/MT/test_cluster_nn_try00_v42.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v43.py ~/data/MT/test_cluster_nn_try00_v43.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v44.py ~/data/MT/test_cluster_nn_try00_v44.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v45.py ~/data/MT/test_cluster_nn_try00_v45.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v46.py ~/data/MT/test_cluster_nn_try00_v46.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v47.py ~/data/MT/test_cluster_nn_try00_v47.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v48.py ~/data/MT/test_cluster_nn_try00_v48.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v49.py ~/data/MT/test_cluster_nn_try00_v49.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v50.py ~/data/MT/test_cluster_nn_try00_v50.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v51.py ~/data/MT/test_cluster_nn_try00_v51.py.log

	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v52.py ~/data/MT/test_cluster_nn_try00_v52.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v53.py ~/data/MT/test_cluster_nn_try00_v53.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v54.py ~/data/MT/test_cluster_nn_try00_v54.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v55.py ~/data/MT/test_cluster_nn_try00_v55.py.log

	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v56.py ~/data/MT/test_cluster_nn_try00_v56.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v57.py ~/data/MT/test_cluster_nn_try00_v57.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v58.py ~/data/MT/test_cluster_nn_try00_v58.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v59.py ~/data/MT/test_cluster_nn_try00_v59.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v60.py ~/data/MT/test_cluster_nn_try00_v60.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v61.py ~/data/MT/test_cluster_nn_try00_v61.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v62.py ~/data/MT/test_cluster_nn_try00_v62.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v63.py ~/data/MT/test_cluster_nn_try00_v63.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v64.py ~/data/MT/test_cluster_nn_try00_v64.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v65.py ~/data/MT/test_cluster_nn_try00_v65.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v66.py ~/data/MT/test_cluster_nn_try00_v66.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v67.py ~/data/MT/test_cluster_nn_try00_v67.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v68.py ~/data/MT/test_cluster_nn_try00_v68.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v69.py ~/data/MT/test_cluster_nn_try00_v69.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v70.py ~/data/MT/test_cluster_nn_try00_v70.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v71.py ~/data/MT/test_cluster_nn_try00_v71.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v72.py ~/data/MT/test_cluster_nn_try00_v72.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v73.py ~/data/MT/test_cluster_nn_try00_v73.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v74.py ~/data/MT/test_cluster_nn_try00_v74.py.log

	# Dateinamen angepasst -> Counter erhöht
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v87.py ~/data/MT/test_cluster_nn_try00_v87.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v88.py ~/data/MT/test_cluster_nn_try00_v88.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v89.py ~/data/MT/test_cluster_nn_try00_v89.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v90.py ~/data/MT/test_cluster_nn_try00_v90.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v91.py ~/data/MT/test_cluster_nn_try00_v91.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v92.py ~/data/MT/test_cluster_nn_try00_v92.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v93.py ~/data/MT/test_cluster_nn_try00_v93.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v94.py ~/data/MT/test_cluster_nn_try00_v94.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v95.py ~/data/MT/test_cluster_nn_try00_v95.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v96.py ~/data/MT/test_cluster_nn_try00_v96.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v97.py ~/data/MT/test_cluster_nn_try00_v97.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v98.py ~/data/MT/test_cluster_nn_try00_v98.py.log

    # Fortsetzen auf dem neuen Cluster
    bash ./scripts/run.sh ./app/test_cluster_nn_try00_v96.py ~/data/MT_gpulab/test_cluster_nn_try00_v96.py.log

	bash ./scripts/run.sh ./app2/test_cluster_nn_try00_v099.py ~/data/MT_gpulab/test_cluster_nn_try00_v099.py.log
	bash ./scripts/run.sh ./app2/test_cluster_nn_try00_v100.py ~/data/MT_gpulab/test_cluster_nn_try00_v100.py.log
	bash ./scripts/run.sh ./app2/test_cluster_nn_try00_v101.py ~/data/MT_gpulab/test_cluster_nn_try00_v101.py.log
	bash ./scripts/run.sh ./app2/test_cluster_nn_try00_v102.py ~/data/MT_gpulab/test_cluster_nn_try00_v102.py.log
	bash ./scripts/run.sh ./app2/test_cluster_nn_try00_v103.py ~/data/MT_gpulab/test_cluster_nn_try00_v103.py.log
	bash ./scripts/run.sh ./app2/test_cluster_nn_try00_v104.py ~/data/MT_gpulab/test_cluster_nn_try00_v104.py.log
	bash ./scripts/run.sh ./app2/test_cluster_nn_try00_v105.py ~/data/MT_gpulab/test_cluster_nn_try00_v105.py.log
	bash ./scripts/run.sh ./app2/test_cluster_nn_try00_v106.py ~/data/MT_gpulab/test_cluster_nn_try00_v106.py.log
	bash ./scripts/run.sh ./app2/test_cluster_nn_try00_v107.py ~/data/MT_gpulab/test_cluster_nn_try00_v107.py.log
	bash ./scripts/run.sh ./app2/test_cluster_nn_try00_v108.py ~/data/MT_gpulab/test_cluster_nn_try00_v108.py.log



    # Copy the modified cudnn layer (see https://github.com/fchollet/keras/issues/8164)
	cp /tmp/test/cudnn_recurrent.py /tmp/test/wrappers.py /opt/conda/lib/python3.5/site-packages/keras/layers/


    # Neuer GPU cluster
    srun --pty --ntasks=1 --cpus-per-task=1 --mem=16G --gres=gpu:1 shifter --image=meierbe8/meierbe8-keras6 bash

    pip install termcolor librosa yattag


