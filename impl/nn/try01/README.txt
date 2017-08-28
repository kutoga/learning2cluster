Basierend auf folgender Beschreibung:

Eine weiter eiterative Möglichkeit wäre folgende:




	clustering(x) {
	    # x is a list of input objects

	    # Create embeddings (dimension reduction)
	    e = create_embeddings(x)

	    # Create an initial state. There is a state per embedding (s is a list)
	    s = create_initial_states(e)

	    # Create an initial global state. This state has a fixed size (independent of the embeddings-count)
	    g = create_global_state(e)

	    # Do 100 iterations (or more general: N, it could vary for the training)
	    for i in 1..100 {

	        # Create a list for the new states
	        states_new = empty_list(length(s))

	        # Go through each embedding
	        for e_i in 0..(length(e)-1) {

	            # Generate a new state for each embedding.
	            # How to do this? Compare each embedding and state with the current
	            # embedding and state (also use the global state). This produces an
	            # output for each input vector. Use a BDLSTM to reduce them to a single vector.
	            # A Dense layer is used to produce the new state with this vector.

	            processed_data = list()
	            for e_j in 1..(length(e)-1) {

	                # Only compare the current embedding with different embeddings (not with itself)
	                if e_i == e_j {
	                    continue
	                }

	                # Do the comparison
	                cmp = compare_data(e[e_i], s[e_i], e[e_j], s[e_j], g)
	                processed_data.append(cmp)
	            }

	            # Reduce the data list to a single vector
	            processed_data_output = BDLSTM(processed_data, return_list=False) # maybe do this somehow else (then move it out to a function)

	            # Create the new state
	            states_new[e_i] = Dense(processed_data_output) # maybe move this to a news function
	        }

	        # Update the states
	        s = states_new

	        # Create a new global state (this is based on a BDLSTM)
	        g = update_global_state(g, concat(s, e)) # maybe remove the g parameter
	    }
	}

	# Do the "normal" softmax classification and get the cluster count from the global state (only from the global state! so we force it to contain somethings useful)
	# Then use the "normal" loss function

Wieso ist diese besser? Tatsächlich ist sie wahrscheinlich schwerer zu trainieren, sie hat jedoch zu den vorherigen Ansätzen den Vorteil, dass sie in jeder Iteration alle Embeddings jeweils mit allen anderen vergleicht. Man hat also bis zu n^2 viele Vergleiche. Diese Komplexität ist (imho?) auch bei vielen Clustering-Algorithmen vorhanden, daher könnte das realistisch sein. Auch speziell ist, dass alle Iterationen "gleich" sind. Die Anzahl Parameter könnte hierdurch stark sinken.