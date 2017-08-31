# MT: Code

Docker auf srv-lab-t-697 starten:
    
    scr 3 # bzw. irgendeine Zahl (erstekllt eine Screen-Session)
    srun --pty --ntasks=1 --cpus-per-task=1 --mem=32G --gres=gpu:1 bash
    nvidia-docker run -it -v ~/data:/data -v ~/code/:/code -v ~/code/tmp:/src/tmp -v ~ keras-meierbe8-3 bash
    pip install termcolor
    cd tmp/ClusterNN
    ln -s /data/MT/ /tmp/test
    bash ./scripts/run.sh ./app/test_minimal_cluster_nn.py /tmp/test_minimal_cluster_nn.py.log

Bzw. falls root & Port-Weiterleitung gewünscht ist:

    nvidia-docker run -it -p 8792:8888 -v ~/data:/data -v ~/code/:/code -v ~/code/tmp:/src/tmp keras-meierbe8-3 bash
    
# Commands
	
	bash ./scripts/run.sh ./app/test_minimal_cluster_nn.py /tmp/test_minimal_cluster_nn.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00.py /tmp/test_cluster_nn_try00.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_noembd.py /tmp/test_cluster_nn_try00_noembd.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try00_fixedc.py /tmp/test_cluster_nn_try00_fixedc.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try01.py /tmp/test_cluster_nn_try01.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try02.py /tmp/test_cluster_nn_try02.py.log
	bash ./scripts/run.sh ./app/test_cluster_nn_try02_mod01.py /tmp/test_cluster_nn_try02_mod01.py.log