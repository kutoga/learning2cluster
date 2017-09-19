Der Clustering-Algorithmus funktioniert ähnlich wie k-means:

    clustering(x) {
        # x is a list of input objects

        # Create embeddings (dimension reduction)
        e = f_create_embeddings(x)

        # Optional: Improve embeddings by the use of a BDLSTM (each layer has own weights)
        for i in range(C_EMBEDDING_ITRS):
            e = BDLSTM(e)

        # For simplicity: Assume the cluster count is fixed to k (this could also be done dynamically)
        k = C_CLUSTER_COUNT

        # Do now a k-means like algorithm

        # Create k initial clusters with a size of EMBEDDING_SIZE: Just use normally distributed data points
        CLUSTERS = norm_rand(k, EMBEDDING_SIZE)

        # Normalize the embeddings. Why? Because the initial cluster centers are also normally distributed
        e = batch_norm(e)

        # Create a probability map for the cluster assignement. It has the size [len(e), k] where [i, j] is the probability
        # that the embedding i should be assigned to the cluster j
        p = zeros(len(e), k)

        # Do now the iterative cluster center re-assignement.
        # A distance function "d" is required. Probably the euclidean distance is a good one.
        for i in range(C_KMEANS_ITRS):

            # Recalculate the assigned cluster centers of the embeddings
            for j in range(len(e)):
                k_dists = []
                for i in range(k)

                    # Use 1 / (0.01 + d(x, y)) to make a function that creates a result for softmax
                    #
                    # Alternative: Use -d(x, y)^2 (why squared? then the euclidean distance calculation is simplified)
                    k_dists.append(1 / (0.01 + d(CLUSTERS[i], e[j])))
                p[j] = softmax(k_dists)

            # Recalculate the cluster centers
            for j in range(k):
                c = 0
                s = 0
                for l in range(len(e)):
                    c += e[l, j] * e[l]
                    s += e[l, j]
                CLUSTERS[j] = c / s

        # The softmaxs are just available in p[embedding]
    }
