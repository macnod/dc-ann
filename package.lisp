;;;; package.lisp

(defpackage :dc-ann
  (:use :cl 
        :cl-ppcre
        :dc-utilities
        :hunchentoot
        :ht-routes
        :sb-thread)
  (:export ann-freeze
           ann-thaw
           anneal
           evaluate-one-hotshot-training
           feed
           id
           log-file
           make-net
           randomize-weights
           run-test
           stop-training
           t-cx
           t-layer
           t-net
           t-neuron
           train
           tset-to-file))
