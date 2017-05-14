;;;; package.lisp

(defpackage :dc-ann
  (:use :cl :dc-utilities :sb-thread :cl-ppcre)
  (:export t-cx t-neuron t-layer t-net randomize-weights feed train
           ann-freeze ann-thaw run-test make-net anneal))
