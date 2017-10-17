
Basierend auf k-means kann man folgendes Clustering aufbauen:

	clustering(x) {
	    # x is a list of input objects

	    # Create embeddings (dimension reduction)
	    e = f_create_embeddings(x)

	    ## Initialize global state
	    ## Also a random intiialization could be used. At least for a part of the global state
	    #g = f_init_global_state(e)

	    # Declare, but do not define a global state (it has a fixed size)
	    g = None

        # Zero initialize the local states
        s = zero_initialized()

	    # Do some iterations (use just a fixed natural number)
	    for i in 1..5 {

	        # The gloabl update state function may use an LSTM (probably it does)
	        g = f_update_global_state(x, s)

	        # Update the local states: Do this for each input independent
	        s = f_update_local_state(x, g)
	    }

	    # Create now two abstract version of the inputs:
	    # - One to count the clusters
	    # - Another to assign each embedding to a cluster

	    #####################
	    # Count the clusters
	    #####################
	    cluster_count = f_cluster_count(g)

	    #####################
	    # Cluster assignement
	    #####################
	    # Do this independent for each local state (do not use an RNN)
	    ca = f_cluster_assignement(s, g)
	}

Änderunge bei Mod01:
- g wird nicht mit 0 initialisiert
- f_update_global_state erhält auch g als Parameter plus am Ende der Funktion wird ein sigmoid gemacht (anstatt RELU)
- f_update_local_state erhält auch s als Parameter plus am Ende der Funktion wird ein sigmoid gemacht (anstatt RELU)
- Die Updates werden jeweils dazuaddiert (und nicht nur zugewiesen)
