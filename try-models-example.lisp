(try-models :ids (range 1 4)
             :topologies '((2 20 10 5 1)
                           (2 100 1)
                           (2 32 8 2 1)
                           (2 30 1))
             :max-iterations-s 1000000
             :randomize-weights-s '(:min -0.5 :max 0.5)
             :thread-count 8
             :training-file "circle-training-data-mini.csv"
             :test-file "circle-test-data-mini.csv")
