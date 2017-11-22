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


Bzw. falls root & Port-Weiterleitung gewünscht ist:

    nvidia-docker run -it -p 8792:8888 -v ~/data:/data -v ~/code/:/code -v ~/code/tmp:/src/tmp keras-meierbe8-3 bash
    
# Commands
	
	bash ./scripts/run.sh ./app/test_minimal_cluster_nn.py /tmp/test/test_minimal_cluster_nn.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00.py /tmp/test/test_cluster_nn_try00.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_noembd.py /tmp/test/test_cluster_nn_try00_noembd.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_noembd_v02.py /tmp/test/test_cluster_nn_try00_noembd_v02.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_fixedc.py /tmp/test/test_cluster_nn_try00_fixedc.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try01.py /tmp/test/test_cluster_nn_try01.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try02.py /tmp/test/test_cluster_nn_try02.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try02_mod01.py /tmp/test/test_cluster_nn_try02_mod01.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try03_kmeans.py /tmp/test/test_cluster_nn_try03_kmeans.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try03_kmeans_v02.py /tmp/test/test_cluster_nn_try03_kmeans_v02.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try03_kmeans_v03.py /tmp/test/test_cluster_nn_try03_kmeans_v03.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try03_kmeans_v04.py /tmp/test/test_cluster_nn_try03_kmeans_v04.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try03_kmeans_v04b.py /tmp/test/test_cluster_nn_try03_kmeans_v04b.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try03_kmeans_v05.py /tmp/test/test_cluster_nn_try03_kmeans_v05.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v02.py /tmp/test/test_cluster_nn_try00_v02.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v03.py /tmp/test/test_cluster_nn_try00_v03.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v04.py /tmp/test/test_cluster_nn_try00_v04.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v05.py /tmp/test/test_cluster_nn_try00_v05.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v06.py /tmp/test/test_cluster_nn_try00_v06.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v07.py /tmp/test/test_cluster_nn_try00_v07.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v08.py /tmp/test/test_cluster_nn_try00_v08.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v09.py /tmp/test/test_cluster_nn_try00_v09.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v10.py /tmp/test/test_cluster_nn_try00_v10.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v11.py /tmp/test/test_cluster_nn_try00_v11.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v12.py /tmp/test/test_cluster_nn_try00_v12.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v13.py /tmp/test/test_cluster_nn_try00_v13.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v14.py /tmp/test/test_cluster_nn_try00_v14.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v15.py /tmp/test/test_cluster_nn_try00_v15.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v16.py /tmp/test/test_cluster_nn_try00_v16.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v17.py /tmp/test/test_cluster_nn_try00_v17.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try04_ddbc.py /tmp/test/test_cluster_nn_try04_ddbc.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v18.py /tmp/test/test_cluster_nn_try00_v18.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try04_ddbc_v02.py /tmp/test/test_cluster_nn_try04_ddbc_v02.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v19.py /tmp/test/test_cluster_nn_try00_v19.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v20.py /tmp/test/test_cluster_nn_try00_v20.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v21.py /tmp/test/test_cluster_nn_try00_v21.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v22.py /tmp/test/test_cluster_nn_try00_v22.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v23.py /tmp/test/test_cluster_nn_try00_v23.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v24.py /tmp/test/test_cluster_nn_try00_v24.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v25.py /tmp/test/test_cluster_nn_try00_v25.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v26.py /tmp/test/test_cluster_nn_try00_v26.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v27.py /tmp/test/test_cluster_nn_try00_v27.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v28.py /tmp/test/test_cluster_nn_try00_v28.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v29.py /tmp/test/test_cluster_nn_try00_v29.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v30.py /tmp/test/test_cluster_nn_try00_v30.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v31.py /tmp/test/test_cluster_nn_try00_v31.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v32.py /tmp/test/test_cluster_nn_try00_v32.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v33.py /tmp/test/test_cluster_nn_try00_v33.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v34.py /tmp/test/test_cluster_nn_try00_v34.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v35.py /tmp/test/test_cluster_nn_try00_v35.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v36.py /tmp/test/test_cluster_nn_try00_v36.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v37.py /tmp/test/test_cluster_nn_try00_v37.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v38.py /tmp/test/test_cluster_nn_try00_v38.py.log
	# From now a keras upgrade may be required (>=2.0.8): pip install https://github.com/fchollet/keras/archive/master.zip
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v39.py /tmp/test/test_cluster_nn_try00_v39.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v40.py /tmp/test/test_cluster_nn_try00_v40.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v41.py /tmp/test/test_cluster_nn_try00_v41.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v42.py /tmp/test/test_cluster_nn_try00_v42.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v43.py /tmp/test/test_cluster_nn_try00_v43.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v44.py /tmp/test/test_cluster_nn_try00_v44.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v45.py /tmp/test/test_cluster_nn_try00_v45.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v46.py /tmp/test/test_cluster_nn_try00_v46.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v47.py /tmp/test/test_cluster_nn_try00_v47.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v48.py /tmp/test/test_cluster_nn_try00_v48.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v49.py /tmp/test/test_cluster_nn_try00_v49.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v50.py /tmp/test/test_cluster_nn_try00_v50.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v51.py /tmp/test/test_cluster_nn_try00_v51.py.log

	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v52.py /tmp/test/test_cluster_nn_try00_v52.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v53.py /tmp/test/test_cluster_nn_try00_v53.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v54.py /tmp/test/test_cluster_nn_try00_v54.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v55.py /tmp/test/test_cluster_nn_try00_v55.py.log

	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v56.py /tmp/test/test_cluster_nn_try00_v56.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v57.py /tmp/test/test_cluster_nn_try00_v57.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v58.py /tmp/test/test_cluster_nn_try00_v58.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v59.py /tmp/test/test_cluster_nn_try00_v59.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v60.py /tmp/test/test_cluster_nn_try00_v60.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v61.py /tmp/test/test_cluster_nn_try00_v61.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v62.py /tmp/test/test_cluster_nn_try00_v62.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v63.py /tmp/test/test_cluster_nn_try00_v63.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v64.py /tmp/test/test_cluster_nn_try00_v64.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v65.py /tmp/test/test_cluster_nn_try00_v65.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v66.py /tmp/test/test_cluster_nn_try00_v66.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v67.py /tmp/test/test_cluster_nn_try00_v67.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v68.py /tmp/test/test_cluster_nn_try00_v68.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v69.py /tmp/test/test_cluster_nn_try00_v69.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v70.py /tmp/test/test_cluster_nn_try00_v70.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v71.py /tmp/test/test_cluster_nn_try00_v71.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v72.py /tmp/test/test_cluster_nn_try00_v72.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v73.py /tmp/test/test_cluster_nn_try00_v73.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v74.py /tmp/test/test_cluster_nn_try00_v74.py.log

	# Dateinamen angepasst -> Counter erhöht
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v87.py /tmp/test/test_cluster_nn_try00_v87.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v88.py /tmp/test/test_cluster_nn_try00_v88.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_v89.py /tmp/test/test_cluster_nn_try00_v89.py.log

    # Copy the modified cudnn layer (see https://github.com/fchollet/keras/issues/8164)
	cp /tmp/test/cudnn_recurrent.py /tmp/test/wrappers.py /opt/conda/lib/python3.5/site-packages/keras/layers/

